"""
Document Intelligence Hub - Streamlit UI
==========================================
Interactive web interface for the Multi-Agent Document Intelligence Hub.

Run with:
    streamlit run streamlit_app.py

Prerequisites:
    pip install streamlit
    pip install -r requirements.txt
    Set GROQ_API_KEY in .env file
"""

import streamlit as st
import json
import time
import os
import sys
import io
from contextlib import redirect_stdout

# Add current directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Document Intelligence Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .agent-card-success {
        border-left-color: #10b981;
    }
    .agent-card-warning {
        border-left-color: #f59e0b;
    }
    .agent-card-error {
        border-left-color: #ef4444;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.2rem;
        color: white;
        text-align: center;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    .score-pass { background-color: #d1fae5; color: #065f46; }
    .score-fail { background-color: #fee2e2; color: #991b1b; }
    .fact-high { border-left: 3px solid #ef4444; padding-left: 0.5rem; }
    .fact-medium { border-left: 3px solid #f59e0b; padding-left: 0.5rem; }
    .fact-low { border-left: 3px solid #3b82f6; padding-left: 0.5rem; }
    .stProgress .st-bo { background-color: #3b82f6; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER: LOAD PIPELINE MODULES
# ============================================================

@st.cache_resource
def load_pipeline_modules():
    """Import pipeline modules once and cache them."""
    try:
        from helpers import (
            load_and_chunk, build_index, search,
            call_llm_cheap, call_llm_strong,
            parse_json, init_state, log_agent, print_log,
            CostTracker, SemanticCache, EvalHarness
        )
        return {
            "load_and_chunk": load_and_chunk,
            "build_index": build_index,
            "search": search,
            "call_llm_cheap": call_llm_cheap,
            "call_llm_strong": call_llm_strong,
            "parse_json": parse_json,
            "init_state": init_state,
            "log_agent": log_agent,
            "CostTracker": CostTracker,
        }
    except Exception as e:
        st.error(f"Failed to load pipeline modules: {e}")
        return None


# ============================================================
# IMPORT AGENT FUNCTIONS
# ============================================================

def get_agent_functions():
    """Import agent functions from the main pipeline."""
    try:
        from project_intelligence_hub import (
            planner, summarizer, fact_extractor,
            quiz_generator, gap_analyzer, critic,
            run_parallel_agents, report_compiler
        )
        return {
            "planner": planner,
            "summarizer": summarizer,
            "fact_extractor": fact_extractor,
            "quiz_generator": quiz_generator,
            "gap_analyzer": gap_analyzer,
            "critic": critic,
            "run_parallel_agents": run_parallel_agents,
            "report_compiler": report_compiler,
        }
    except Exception as e:
        st.error(f"Failed to load agent functions: {e}")
        return None


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render the sidebar with settings and file upload."""
    with st.sidebar:
        st.markdown("### Settings")

        # API Key check
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key and api_key != "your-groq-key-here":
            st.success("Groq API Key: Connected")
        else:
            st.warning("Groq API Key: Not set")
            st.caption("Set `GROQ_API_KEY` in `.env` file")
            manual_key = st.text_input("Or enter API key:", type="password")
            if manual_key:
                os.environ["GROQ_API_KEY"] = manual_key
                st.rerun()

        st.markdown("---")

        # Pipeline settings
        st.markdown("### Pipeline Config")
        max_retries = st.slider("Max Retries", 0, 3, 1,
                                help="Number of retry attempts if quality score < 0.7")
        budget = st.slider("Budget (USD)", 0.05, 1.00, 0.10, 0.05,
                           help="Cost budget limit (Groq free tier = $0.00)")
        parallel = st.checkbox("Run agents in parallel", value=True,
                               help="Run 4 analysis agents concurrently")

        st.markdown("---")

        # Rate limit info
        st.markdown("### Groq Free Tier Limits")
        st.caption("""
        **llama-3.1-8b-instant** (used for all calls)
        - 30 RPM | 14.4K RPD | 6K TPM | 500K TPD

        Using 8b only to conserve free tier budget.
        Rate limiting: 5s delay between calls.
        If you hit 429 errors, auto-retry after 30s.
        """)

        st.markdown("---")

        # Architecture diagram
        with st.expander("Architecture"):
            st.code("""
Planner (8B)
    |
    v
[Summarizer | Facts | Quiz | Gaps]  (8B, parallel)
    |
    v
Critic (8B, quality gate)
    |
    v  (retry if score < 0.7)
Report Compiler
            """, language="text")

        return {
            "max_retries": max_retries,
            "budget": budget,
            "parallel": parallel,
        }


# ============================================================
# FILE UPLOAD & PROCESSING
# ============================================================

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temp location and return the path."""
    import tempfile
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# ============================================================
# PIPELINE EXECUTION UI
# ============================================================

def run_pipeline_with_ui(pdf_path, config, modules, agents):
    """Run the full pipeline with Streamlit progress indicators."""

    # Initialize tracking
    tracker = modules["CostTracker"](budget=config["budget"])
    total_steps = 7
    progress_bar = st.progress(0, text="Starting pipeline...")

    results = {}
    timing = {}

    # ----- Step 1: Load & Chunk -----
    progress_bar.progress(1 / total_steps, text="[1/7] Loading and chunking PDF...")
    step_start = time.time()
    with st.spinner("Extracting text from PDF..."):
        chunks = modules["load_and_chunk"](pdf_path)
    timing["load"] = time.time() - step_start
    results["total_chunks"] = len(chunks)
    st.success(f"Loaded {len(chunks)} chunks from PDF")

    # ----- Step 2: Build Index -----
    progress_bar.progress(2 / total_steps, text="[2/7] Building semantic index...")
    step_start = time.time()
    with st.spinner("Embedding chunks (local model, no API calls)..."):
        index = modules["build_index"](chunks)
    timing["index"] = time.time() - step_start

    # ----- Step 3: Select Diverse Chunks -----
    progress_bar.progress(3 / total_steps, text="[3/7] Selecting diverse chunks...")
    state = modules["init_state"]()
    all_results = []
    for q in ["main topic overview", "key concept definition", "important conclusion"]:
        all_results.extend(modules["search"](index, chunks, q, k=3))
    seen = set()
    state["chunks"] = [c for c in all_results
                       if c["index"] not in seen and not seen.add(c["index"])][:10]
    results["selected_chunks"] = len(state["chunks"])

    # ----- Step 4: Planner -----
    progress_bar.progress(4 / total_steps, text="[4/7] Running Planner agent...")
    step_start = time.time()
    with st.spinner("Planner analyzing document structure..."):
        state = agents["planner"](state, tracker)
    timing["planner"] = time.time() - step_start
    results["plan"] = state.get("plan", {})

    # ----- Step 5: Analysis Agents -----
    progress_bar.progress(5 / total_steps, text="[5/7] Running analysis agents...")
    step_start = time.time()

    if config["parallel"]:
        with st.spinner("Running 4 agents in parallel (Summarizer, Facts, Quiz, Gaps)..."):
            state = agents["run_parallel_agents"](state, tracker)
    else:
        agent_names = [
            ("summarizer", "Summarizer"),
            ("fact_extractor", "Fact Extractor"),
            ("quiz_generator", "Quiz Generator"),
            ("gap_analyzer", "Gap Analyzer"),
        ]
        for fn_name, display_name in agent_names:
            with st.spinner(f"Running {display_name}..."):
                state = agents[fn_name](state, tracker)

    timing["analysis"] = time.time() - step_start

    results["summary"] = state.get("summary", {})
    results["facts"] = state.get("facts", {})
    results["quiz"] = state.get("quiz", {})
    results["gaps"] = state.get("gaps", {})

    # ----- Step 6: Critic -----
    progress_bar.progress(6 / total_steps, text="[6/7] Running Critic (quality gate)...")
    step_start = time.time()
    with st.spinner("Critic validating outputs against source..."):
        state = agents["critic"](state, tracker)
    timing["critic"] = time.time() - step_start
    results["critic"] = state.get("critic", {})
    results["critic_score"] = state.get("critic_score", 0)

    # ----- Adaptive Retry -----
    retry_count = 0
    while results["critic_score"] < 0.7 and retry_count < config["max_retries"]:
        retry_count += 1
        progress_bar.progress(6 / total_steps,
                              text=f"[6/7] Retrying (attempt {retry_count + 1})...")
        with st.spinner(f"Adaptive retry {retry_count}: re-planning with critic feedback..."):
            try:
                tracker.check_budget()
                state = agents["planner"](state, tracker)
                if config["parallel"]:
                    state = agents["run_parallel_agents"](state, tracker)
                else:
                    for fn_name in ["summarizer", "fact_extractor", "quiz_generator", "gap_analyzer"]:
                        state = agents[fn_name](state, tracker)
                state = agents["critic"](state, tracker)
                results["summary"] = state.get("summary", {})
                results["facts"] = state.get("facts", {})
                results["quiz"] = state.get("quiz", {})
                results["gaps"] = state.get("gaps", {})
                results["critic"] = state.get("critic", {})
                results["critic_score"] = state.get("critic_score", 0)
            except RuntimeError as e:
                st.warning(f"Budget exceeded during retry: {e}")
                break

    results["retry_count"] = retry_count

    # ----- Step 7: Report -----
    progress_bar.progress(7 / total_steps, text="[7/7] Compiling report...")
    state = agents["report_compiler"](state)
    results["report"] = state.get("report", "")
    results["state"] = state
    results["timing"] = timing
    results["tracker"] = tracker

    progress_bar.progress(1.0, text="Pipeline complete!")
    time.sleep(0.5)
    progress_bar.empty()

    return results


# ============================================================
# RESULTS DISPLAY
# ============================================================

def display_results(results):
    """Display pipeline results in a nice UI."""

    # ----- Top Metrics Row -----
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)

    critic_score = results.get("critic_score", 0)
    verdict = "PASS" if critic_score >= 0.7 else "FAIL"
    verdict_color = "green" if critic_score >= 0.7 else "red"

    with col1:
        st.metric("Quality Score", f"{critic_score:.2f}/1.0")
    with col2:
        st.metric("Verdict", verdict)
    with col3:
        st.metric("Retries", results.get("retry_count", 0))
    with col4:
        plan = results.get("plan", {})
        st.metric("Complexity", plan.get("estimated_complexity", "?").title())
    with col5:
        st.metric("Chunks Used", results.get("selected_chunks", "?"))

    # ----- Tabs for Each Section -----
    tab_summary, tab_facts, tab_quiz, tab_gaps, tab_quality, tab_report, tab_costs = st.tabs([
        "Executive Summary",
        "Key Facts",
        "Practice Quiz",
        "Gap Analysis",
        "Quality Validation",
        "Full Report",
        "Cost & Timing",
    ])

    # ---- TAB: Executive Summary ----
    with tab_summary:
        plan = results.get("plan", {})
        summary_data = results.get("summary", {})

        # Document metadata
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"**Document Type:** {plan.get('document_type', 'Unknown')}")
        with col_b:
            st.markdown(f"**Target Audience:** {plan.get('target_audience', 'Unknown')}")
        with col_c:
            st.markdown(f"**Complexity:** {plan.get('estimated_complexity', 'Unknown')}")

        st.markdown("#### Main Topic")
        st.info(plan.get("main_topic", "Not identified"))

        st.markdown("#### Key Themes")
        themes = plan.get("key_themes", [])
        if themes:
            cols = st.columns(min(len(themes), 5))
            for i, theme in enumerate(themes):
                with cols[i % len(cols)]:
                    st.markdown(f"- {theme}")

        st.markdown("---")
        st.markdown("#### Executive Summary")
        st.write(summary_data.get("summary", "Summary not generated."))

        key_points = summary_data.get("key_points", [])
        if key_points:
            st.markdown("#### Key Points")
            for i, point in enumerate(key_points, 1):
                st.markdown(f"{i}. {point}")

    # ---- TAB: Key Facts ----
    with tab_facts:
        facts_data = results.get("facts", {})
        facts_list = facts_data.get("facts", [])

        if not facts_list:
            st.warning("No facts extracted.")
        else:
            st.markdown(f"**{len(facts_list)} facts extracted**")

            # Group by importance
            for importance in ["high", "medium", "low"]:
                imp_facts = [f for f in facts_list if f.get("importance", "").lower() == importance]
                if imp_facts:
                    emoji = {"high": "🔴", "medium": "🟡", "low": "🔵"}[importance]
                    st.markdown(f"#### {emoji} {importance.title()} Importance")
                    for fact in imp_facts:
                        fact_type = fact.get("type", "fact").upper()
                        st.markdown(f"- **[{fact_type}]** {fact.get('fact', '')}")

    # ---- TAB: Practice Quiz ----
    with tab_quiz:
        quiz_data = results.get("quiz", {})
        questions = quiz_data.get("questions", [])

        if not questions:
            st.warning("No quiz questions generated.")
        else:
            st.markdown(f"**{len(questions)} questions generated**")

            for i, q in enumerate(questions, 1):
                difficulty = q.get("difficulty", "medium")
                diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(difficulty, "⚪")

                with st.expander(f"Q{i} {diff_emoji} [{difficulty}]: {q.get('question', '')}", expanded=(i <= 2)):
                    options = q.get("options", {})
                    correct = q.get("correct", "")

                    # Show options as radio buttons
                    option_list = [f"{letter}) {text}" for letter, text in sorted(options.items())]
                    if option_list:
                        # Create a unique key for each question
                        selected = st.radio(
                            "Select your answer:",
                            option_list,
                            key=f"quiz_q{i}",
                            label_visibility="collapsed"
                        )

                        # Check answer button
                        if st.button(f"Check Answer", key=f"check_q{i}"):
                            selected_letter = selected[0] if selected else ""
                            if selected_letter == correct:
                                st.success(f"Correct! {correct}) is right.")
                            else:
                                st.error(f"Incorrect. The correct answer is {correct}).")
                            st.info(f"**Explanation:** {q.get('explanation', 'No explanation provided.')}")

    # ---- TAB: Gap Analysis ----
    with tab_gaps:
        gaps_data = results.get("gaps", {})
        gaps_list = gaps_data.get("gaps", [])
        coverage = gaps_data.get("coverage_score", "?")

        st.metric("Coverage Score", f"{coverage}/1.0" if isinstance(coverage, (int, float)) else coverage)

        if gaps_data.get("recommendation"):
            st.info(f"**Recommendation:** {gaps_data['recommendation']}")

        if not gaps_list:
            st.success("No significant gaps identified!")
        else:
            for i, gap in enumerate(gaps_list, 1):
                severity = gap.get("severity", "moderate").lower()
                sev_emoji = {"critical": "🔴", "moderate": "🟡", "minor": "🔵"}.get(severity, "⚪")
                sev_color = {"critical": "error", "moderate": "warning", "minor": "info"}.get(severity, "info")

                with st.container():
                    st.markdown(f"#### {sev_emoji} Gap {i}: {gap.get('topic', 'Unknown')}")
                    st.markdown(f"**Severity:** {severity.title()}")
                    st.markdown(f"**Why important:** {gap.get('why_important', '')}")
                    st.markdown("---")

    # ---- TAB: Quality Validation ----
    with tab_quality:
        critic_data = results.get("critic", {})
        scores = critic_data.get("scores", {})

        st.markdown(f"### Overall Score: {results.get('critic_score', 0):.2f}/1.0")

        if critic_score >= 0.7:
            st.success(f"**PASSED** - Quality threshold met (>= 0.7)")
        else:
            st.error(f"**FAILED** - Below quality threshold (< 0.7)")

        if scores:
            st.markdown("#### Per-Section Scores")
            cols = st.columns(len(scores))
            for i, (section, data) in enumerate(scores.items()):
                with cols[i]:
                    score = data.get("score", 0)
                    st.metric(section.title(), f"{score:.2f}")
                    issues = data.get("issues", [])
                    if issues:
                        for issue in issues:
                            st.caption(f"- {issue}")

        critical_issues = critic_data.get("critical_issues", [])
        if critical_issues:
            st.markdown("#### Critical Issues")
            for issue in critical_issues:
                st.error(issue)

        hints = critic_data.get("improvement_hints", [])
        if hints:
            st.markdown("#### Improvement Hints")
            for hint in hints:
                st.info(hint)

    # ---- TAB: Full Report ----
    with tab_report:
        report_text = results.get("report", "No report generated.")
        st.code(report_text, language="text")

        st.download_button(
            label="Download Report (TXT)",
            data=report_text,
            file_name="intelligence_hub_report.txt",
            mime="text/plain"
        )

    # ---- TAB: Costs & Timing ----
    with tab_costs:
        tracker = results.get("tracker")
        timing = results.get("timing", {})

        col_c1, col_c2, col_c3 = st.columns(3)
        total_time = sum(timing.values())

        with col_c1:
            st.metric("Total Time", f"{total_time:.1f}s")
        with col_c2:
            if tracker:
                st.metric("Total Cost", f"${tracker.total_cost:.4f}")
            else:
                st.metric("Total Cost", "$0.00")
        with col_c3:
            if tracker:
                total_tokens = tracker.total_input_tokens + tracker.total_output_tokens
                st.metric("Total Tokens", f"{total_tokens:,}")

        # Timing breakdown
        st.markdown("#### Timing Breakdown")
        timing_data = {
            "Load & Chunk": timing.get("load", 0),
            "Build Index": timing.get("index", 0),
            "Planner": timing.get("planner", 0),
            "Analysis Agents": timing.get("analysis", 0),
            "Critic": timing.get("critic", 0),
        }
        for stage, t in timing_data.items():
            pct = (t / total_time * 100) if total_time > 0 else 0
            st.markdown(f"**{stage}:** {t:.1f}s ({pct:.0f}%)")
            st.progress(min(pct / 100, 1.0))

        # Per-agent cost breakdown
        if tracker and tracker.calls:
            st.markdown("#### Cost Breakdown by Agent")
            agent_costs = {}
            for call in tracker.calls:
                name = call["agent"]
                if name not in agent_costs:
                    agent_costs[name] = {"calls": 0, "tokens": 0, "cost": 0, "latency": 0}
                agent_costs[name]["calls"] += 1
                agent_costs[name]["tokens"] += call["input_tokens"] + call["output_tokens"]
                agent_costs[name]["cost"] += call["cost"]
                agent_costs[name]["latency"] += call["latency_ms"]

            # Display as a table
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Agent": name,
                    "Calls": data["calls"],
                    "Tokens": data["tokens"],
                    "Cost ($)": f"{data['cost']:.4f}",
                    "Latency (ms)": data["latency"],
                    "Model": next((c["model"] for c in tracker.calls if c["agent"] == name), "?")
                }
                for name, data in sorted(agent_costs.items(), key=lambda x: -x[1]["tokens"])
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown(f"""
            **Budget:** ${tracker.budget:.2f} |
            **Spent:** ${tracker.total_cost:.4f} |
            **Remaining:** ${tracker.remaining():.4f}
            """)


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown('<div class="main-header">Document Intelligence Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Agent PDF Analysis powered by Groq (Free Tier)</div>',
                unsafe_allow_html=True)

    # Sidebar
    config = render_sidebar()

    # Load modules
    modules = load_pipeline_modules()
    agent_fns = get_agent_functions()

    if not modules or not agent_fns:
        st.error("Failed to load pipeline. Check that all dependencies are installed and GROQ_API_KEY is set.")
        st.code("pip install -r requirements.txt", language="bash")
        return

    # ---- Main Content Area ----
    st.markdown("### Upload a PDF Document")

    # File upload
    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload any PDF document for analysis"
        )

    with col_sample:
        st.markdown("**Or use a sample:**")
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")
        sample_files = []
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith(".pdf")]

        if sample_files:
            selected_sample = st.selectbox("Sample documents", ["-- Select --"] + sample_files)
        else:
            selected_sample = "-- Select --"
            st.caption("No sample docs found in `sample_docs/`")

    # Determine which file to use
    pdf_path = None
    if uploaded_file is not None:
        pdf_path = save_uploaded_file(uploaded_file)
        st.info(f"Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.0f} KB)")
    elif selected_sample and selected_sample != "-- Select --":
        pdf_path = os.path.join(sample_dir, selected_sample)
        st.info(f"Using sample: **{selected_sample}**")

    # Run button
    if pdf_path:
        st.markdown("---")

        col_run, col_info = st.columns([1, 3])
        with col_run:
            run_clicked = st.button("Analyze Document", type="primary", use_container_width=True)
        with col_info:
            st.caption(
                f"Config: retries={config['max_retries']}, budget=${config['budget']:.2f}, "
                f"parallel={'Yes' if config['parallel'] else 'No'} | "
                f"Rate limit: 4s delay between API calls"
            )

        if run_clicked:
            try:
                results = run_pipeline_with_ui(pdf_path, config, modules, agent_fns)
                display_results(results)
            except RuntimeError as e:
                if "budget" in str(e).lower():
                    st.error(f"Budget exceeded: {e}")
                elif "api" in str(e).lower() or "key" in str(e).lower():
                    st.error(f"API Error: {e}")
                else:
                    st.error(f"Pipeline error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.exception(e)

    # Display previous results if they exist in session state
    elif "last_results" in st.session_state:
        display_results(st.session_state["last_results"])

    # ---- Footer ----
    st.markdown("---")
    st.caption(
        "Built with Streamlit | Powered by Groq (Free Tier) | "
        "Embeddings: all-MiniLM-L6-v2 (local) | Vector Search: FAISS"
    )


if __name__ == "__main__":
    main()
