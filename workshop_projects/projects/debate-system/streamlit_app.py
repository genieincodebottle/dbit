"""
Multi-Agent Debate System - Streamlit UI
==========================================
Interactive web interface for the Multi-Agent Debate System.

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

# Add current directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Debate System",
    page_icon="\u2696\ufe0f",
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
    .for-card {
        background-color: #f0fdf4;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #22c55e;
        margin-bottom: 1rem;
    }
    .against-card {
        background-color: #fef2f2;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #ef4444;
        margin-bottom: 1rem;
    }
    .judge-card {
        background-color: #fefce8;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #eab308;
        margin-bottom: 1rem;
    }
    .synthesis-card {
        background-color: #eff6ff;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD PIPELINE MODULES
# ============================================================

@st.cache_resource
def load_pipeline_modules():
    """Import pipeline modules once and cache them."""
    try:
        from helpers import (
            load_and_chunk, build_index, search,
            call_llm_cheap, call_llm_strong,
            parse_json, init_state, log_agent, print_log,
            CostTracker
        )
        from project_debate_system import (
            debate_planner, researcher, debater,
            cross_examiner, judge, synthesizer,
            run_parallel, format_report
        )
        return {
            "load_and_chunk": load_and_chunk,
            "build_index": build_index,
            "search": search,
            "init_state": init_state,
            "CostTracker": CostTracker,
            "debate_planner": debate_planner,
            "researcher": researcher,
            "debater": debater,
            "cross_examiner": cross_examiner,
            "judge": judge,
            "synthesizer": synthesizer,
            "run_parallel": run_parallel,
            "format_report": format_report,
        }
    except Exception as e:
        st.error(f"Failed to load pipeline modules: {e}")
        return None


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render the sidebar with settings."""
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

        # Debate settings
        st.markdown("### Debate Config")
        max_rounds = st.slider("Max Rounds", 1, 3, 2,
                               help="Extra rounds trigger cross-examination if the verdict is close")
        budget = st.slider("Budget (USD)", 0.05, 1.00, 0.30, 0.05,
                           help="Cost budget limit (Groq free tier = $0.00)")

        st.markdown("---")

        # Rate limit info
        st.markdown("### Groq Free Tier Limits")
        st.caption("""
        **llama-3.1-8b-instant** (used for all calls)
        - 30 RPM | 14.4K RPD | 6K TPM | 500K TPD

        Using 8b only to conserve free tier budget.
        Rate limiting: 5s delay between calls.
        Students can swap to 70b in helpers.py
        if they have a paid plan.
        """)

        st.markdown("---")

        # Architecture diagram
        with st.expander("Architecture"):
            st.code("""
Round 1:
  Planner (8B)
      |
  [Researcher FOR | Researcher AGAINST]
      |
  [Debater FOR | Debater AGAINST]  (8B)
      |
  Judge (8B)

Round 2 (if close):
  [Cross-Exam FOR | Cross-Exam AGAINST]
      |
  Judge (8B, final)

Synthesis:
  Synthesizer (8B) -> Report
            """, language="text")

        return {
            "max_rounds": max_rounds,
            "budget": budget,
        }


# ============================================================
# FILE UPLOAD
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
# PIPELINE EXECUTION
# ============================================================

def run_debate_with_ui(pdf_path, topic, config, modules):
    """Run the full debate pipeline with Streamlit progress indicators."""

    tracker = modules["CostTracker"](budget=config["budget"])
    timing = {}

    # Step 1: Load & Chunk
    progress = st.progress(0, text="[1/8] Loading PDF...")
    step_start = time.time()
    with st.spinner("Extracting text from PDF..."):
        chunks = modules["load_and_chunk"](pdf_path)
    timing["load"] = time.time() - step_start
    st.success(f"Loaded {len(chunks)} chunks from PDF")

    # Step 2: Build Index
    progress.progress(1/8, text="[2/8] Building semantic index...")
    step_start = time.time()
    with st.spinner("Embedding chunks (local model, no API calls)..."):
        index = modules["build_index"](chunks)
    timing["index"] = time.time() - step_start

    # Initialize state
    state = modules["init_state"](topic)
    state["topic"] = topic
    state["pdf_path"] = pdf_path
    state["_index"] = index
    state["_all_chunks"] = chunks
    state["chunks"] = modules["search"](index, chunks, topic, k=8)

    # Step 3: Plan
    progress.progress(2/8, text="[3/8] Planning debate framework...")
    step_start = time.time()
    with st.spinner("Planner analyzing topic and document..."):
        state = modules["debate_planner"](state, tracker)
        tracker.check_budget()
    timing["planner"] = time.time() - step_start

    # Step 4: Research (parallel)
    progress.progress(3/8, text="[4/8] Gathering evidence (FOR & AGAINST)...")
    step_start = time.time()
    with st.spinner("Researchers gathering evidence for both sides..."):
        state = modules["run_parallel"](
            [(modules["researcher"], "for"), (modules["researcher"], "against")],
            state, tracker
        )
        tracker.check_budget()
    timing["research"] = time.time() - step_start

    # Step 5: Debate (parallel)
    progress.progress(4/8, text="[5/8] Building arguments (FOR & AGAINST)...")
    step_start = time.time()
    with st.spinner("Debaters building arguments..."):
        state = modules["run_parallel"](
            [(modules["debater"], "for"), (modules["debater"], "against")],
            state, tracker
        )
        tracker.check_budget()
    timing["debate"] = time.time() - step_start

    # Step 6: Judge Round 1
    progress.progress(5/8, text="[6/8] Judge evaluating Round 1...")
    step_start = time.time()
    with st.spinner("Judge scoring both arguments..."):
        state = modules["judge"](state, tracker, round_num=1)
    state["rounds_played"] = 1
    timing["judge_r1"] = time.time() - step_start

    # Step 7: Cross-examination (if close)
    margin = state.get("judgment", {}).get("margin", "decisive")
    if config["max_rounds"] > 1 and margin in ("razor-thin", "narrow") and tracker.remaining() > 0.15:
        progress.progress(6/8, text=f"[7/8] Cross-examination (debate is {margin})...")
        step_start = time.time()
        with st.spinner("Cross-examiners challenging opponents..."):
            state = modules["run_parallel"](
                [(modules["cross_examiner"], "for"), (modules["cross_examiner"], "against")],
                state, tracker
            )
            tracker.check_budget()

        with st.spinner("Judge re-evaluating with cross-examination..."):
            state = modules["judge"](state, tracker, round_num=2)
            state["rounds_played"] = 2
        timing["cross_exam"] = time.time() - step_start
    else:
        if margin not in ("razor-thin", "narrow"):
            st.info(f"Skipping cross-examination: result was {margin}")
        elif tracker.remaining() <= 0.15:
            st.info(f"Skipping cross-examination: budget too low (${tracker.remaining():.4f})")

    # Step 8: Synthesis
    progress.progress(7/8, text="[8/8] Synthesizing balanced analysis...")
    step_start = time.time()
    with st.spinner("Synthesizer writing balanced analysis..."):
        state = modules["synthesizer"](state, tracker)
    timing["synthesis"] = time.time() - step_start

    # Format report
    report = modules["format_report"](state)
    state["report"] = report

    progress.progress(1.0, text="Debate complete!")
    time.sleep(0.5)
    progress.empty()

    return {
        "state": state,
        "tracker": tracker,
        "timing": timing,
        "report": report,
    }


# ============================================================
# RESULTS DISPLAY
# ============================================================

def display_results(results):
    """Display debate results in a rich UI."""

    state = results["state"]
    tracker = results["tracker"]
    timing = results["timing"]
    judgment = state.get("judgment", {})
    plan = state.get("debate_plan", {})

    # Top metrics row
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)

    f_total = judgment.get("for_score", {}).get("total", 0)
    a_total = judgment.get("against_score", {}).get("total", 0)
    winner = judgment.get("winner", "?").upper()
    margin = judgment.get("margin", "?")

    with col1:
        st.metric("FOR Score", f"{f_total}/50")
    with col2:
        st.metric("AGAINST Score", f"{a_total}/50")
    with col3:
        st.metric("Winner", winner)
    with col4:
        st.metric("Margin", margin.title())
    with col5:
        st.metric("Rounds", state.get("rounds_played", 1))

    # Tabs
    tab_overview, tab_for, tab_against, tab_cross, tab_verdict, tab_synthesis, tab_report, tab_costs = st.tabs([
        "Debate Framework",
        "FOR Argument",
        "AGAINST Argument",
        "Cross-Examination",
        "Judge's Verdict",
        "Balanced Synthesis",
        "Full Report",
        "Cost & Timing",
    ])

    # ---- TAB: Debate Framework ----
    with tab_overview:
        st.markdown("#### Topic")
        st.info(state.get("topic", "?"))

        col_f, col_a = st.columns(2)
        with col_f:
            st.markdown('<div class="for-card">', unsafe_allow_html=True)
            st.markdown(f"**FOR Position:**\n\n{plan.get('for_position', '?')}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_a:
            st.markdown('<div class="against-card">', unsafe_allow_html=True)
            st.markdown(f"**AGAINST Position:**\n\n{plan.get('against_position', '?')}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### Debate Dimensions")
        dims = plan.get("dimensions", [])
        if dims:
            cols = st.columns(min(len(dims), 4))
            for i, dim in enumerate(dims):
                with cols[i % len(cols)]:
                    st.markdown(f"- {dim}")

        if plan.get("context_from_document"):
            with st.expander("Document Context"):
                st.write(plan["context_from_document"])

    # ---- TAB: FOR Argument ----
    with tab_for:
        arg_for = state.get("argument_for", {})
        evidence_for = state.get("evidence_for", [])

        st.markdown('<div class="for-card">', unsafe_allow_html=True)
        st.markdown(f"**Thesis:** \"{arg_for.get('opening_statement', '?')}\"")
        st.markdown('</div>', unsafe_allow_html=True)

        arguments = arg_for.get("arguments", [])
        for i, a in enumerate(arguments, 1):
            with st.expander(f"Argument {i}: {a.get('point', '?')}", expanded=True):
                st.markdown(f"**Evidence:** \"{a.get('evidence', '?')}\"")
                st.markdown(f"**Reasoning:** {a.get('reasoning', '?')}")

        if arg_for.get("counter_to_opposition"):
            st.markdown("#### Preemptive Rebuttal")
            st.write(arg_for["counter_to_opposition"])

        if arg_for.get("closing_statement"):
            st.markdown("#### Closing Statement")
            st.success(arg_for["closing_statement"])

        with st.expander(f"Evidence Chunks ({len(evidence_for)} chunks)"):
            for e in evidence_for:
                st.caption(f"[Chunk {e.get('index', '?')} | Score: {e.get('score', '?')}]")
                st.text(e.get("text", "")[:200] + "...")

    # ---- TAB: AGAINST Argument ----
    with tab_against:
        arg_against = state.get("argument_against", {})
        evidence_against = state.get("evidence_against", [])

        st.markdown('<div class="against-card">', unsafe_allow_html=True)
        st.markdown(f"**Thesis:** \"{arg_against.get('opening_statement', '?')}\"")
        st.markdown('</div>', unsafe_allow_html=True)

        arguments = arg_against.get("arguments", [])
        for i, a in enumerate(arguments, 1):
            with st.expander(f"Argument {i}: {a.get('point', '?')}", expanded=True):
                st.markdown(f"**Evidence:** \"{a.get('evidence', '?')}\"")
                st.markdown(f"**Reasoning:** {a.get('reasoning', '?')}")

        if arg_against.get("counter_to_opposition"):
            st.markdown("#### Preemptive Rebuttal")
            st.write(arg_against["counter_to_opposition"])

        if arg_against.get("closing_statement"):
            st.markdown("#### Closing Statement")
            st.error(arg_against["closing_statement"])

        with st.expander(f"Evidence Chunks ({len(evidence_against)} chunks)"):
            for e in evidence_against:
                st.caption(f"[Chunk {e.get('index', '?')} | Score: {e.get('score', '?')}]")
                st.text(e.get("text", "")[:200] + "...")

    # ---- TAB: Cross-Examination ----
    with tab_cross:
        cx_for = state.get("cross_exam_for")
        cx_against = state.get("cross_exam_against")

        if not cx_for and not cx_against:
            st.info("No cross-examination round was triggered. This happens when the initial verdict is decisive.")
        else:
            col_cx_f, col_cx_a = st.columns(2)

            with col_cx_f:
                st.markdown("#### FOR challenges AGAINST")
                if cx_for:
                    st.markdown(f"**Weakest Point Identified:** {cx_for.get('weakest_point', '?')}")
                    st.markdown(f"**Challenge:** {cx_for.get('challenge', '?')}")
                    questions = cx_for.get("unanswerable_questions", [])
                    if questions:
                        st.markdown("**Unanswerable Questions:**")
                        for q in questions:
                            st.markdown(f"- {q}")
                    if cx_for.get("additional_evidence"):
                        st.markdown(f"**Additional Evidence:** {cx_for['additional_evidence']}")

            with col_cx_a:
                st.markdown("#### AGAINST challenges FOR")
                if cx_against:
                    st.markdown(f"**Weakest Point Identified:** {cx_against.get('weakest_point', '?')}")
                    st.markdown(f"**Challenge:** {cx_against.get('challenge', '?')}")
                    questions = cx_against.get("unanswerable_questions", [])
                    if questions:
                        st.markdown("**Unanswerable Questions:**")
                        for q in questions:
                            st.markdown(f"- {q}")
                    if cx_against.get("additional_evidence"):
                        st.markdown(f"**Additional Evidence:** {cx_against['additional_evidence']}")

    # ---- TAB: Judge's Verdict ----
    with tab_verdict:
        st.markdown('<div class="judge-card">', unsafe_allow_html=True)
        st.markdown(f"**Winner: {winner}** ({margin})")
        st.markdown(f"\n{judgment.get('reasoning', '?')}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Score breakdown table
        st.markdown("#### Score Breakdown")
        f_s = judgment.get("for_score", {})
        a_s = judgment.get("against_score", {})

        criteria = ["evidence_quality", "logical_coherence", "completeness", "persuasiveness", "honesty"]
        score_data = []
        for c in criteria:
            score_data.append({
                "Criterion": c.replace("_", " ").title(),
                "FOR": f"{f_s.get(c, '?')}/10",
                "AGAINST": f"{a_s.get(c, '?')}/10",
            })
        score_data.append({
            "Criterion": "TOTAL",
            "FOR": f"{f_s.get('total', '?')}/50",
            "AGAINST": f"{a_s.get('total', '?')}/50",
        })

        import pandas as pd
        df = pd.DataFrame(score_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        col_sp, col_wp = st.columns(2)
        with col_sp:
            st.markdown("#### Strongest Points")
            st.success(f"**FOR:** {judgment.get('strongest_point_for', '?')}")
            st.error(f"**AGAINST:** {judgment.get('strongest_point_against', '?')}")
        with col_wp:
            st.markdown("#### Weakest Points")
            st.warning(f"**FOR:** {judgment.get('weakest_point_for', '?')}")
            st.warning(f"**AGAINST:** {judgment.get('weakest_point_against', '?')}")

    # ---- TAB: Balanced Synthesis ----
    with tab_synthesis:
        synthesis = state.get("synthesis", {})

        st.markdown('<div class="synthesis-card">', unsafe_allow_html=True)
        st.markdown("#### Balanced Analysis")
        st.write(synthesis.get("balanced_analysis", "No synthesis generated."))
        st.markdown('</div>', unsafe_allow_html=True)

        common = synthesis.get("common_ground", [])
        if common:
            st.markdown("#### Common Ground")
            for p in common:
                st.markdown(f"- {p}")

        if synthesis.get("key_tension"):
            st.markdown("#### Key Tension")
            st.warning(synthesis["key_tension"])

        if synthesis.get("nuanced_conclusion"):
            st.markdown("#### Nuanced Conclusion")
            st.info(synthesis["nuanced_conclusion"])

    # ---- TAB: Full Report ----
    with tab_report:
        report_text = results.get("report", "No report generated.")
        st.code(report_text, language="text")

        st.download_button(
            label="Download Report (TXT)",
            data=report_text,
            file_name="debate_report.txt",
            mime="text/plain"
        )

    # ---- TAB: Cost & Timing ----
    with tab_costs:
        col_c1, col_c2, col_c3 = st.columns(3)
        total_time = sum(timing.values())

        with col_c1:
            st.metric("Total Time", f"{total_time:.1f}s")
        with col_c2:
            st.metric("Total Cost", f"${tracker.total_cost:.4f}")
        with col_c3:
            total_tokens = tracker.total_input_tokens + tracker.total_output_tokens
            st.metric("Total Tokens", f"{total_tokens:,}")

        # Timing breakdown
        st.markdown("#### Timing Breakdown")
        timing_labels = {
            "load": "Load & Chunk",
            "index": "Build Index",
            "planner": "Planner",
            "research": "Research (parallel)",
            "debate": "Debaters (parallel)",
            "judge_r1": "Judge Round 1",
            "cross_exam": "Cross-Examination + Judge R2",
            "synthesis": "Synthesizer",
        }
        for key, label in timing_labels.items():
            t = timing.get(key, 0)
            if t > 0:
                pct = (t / total_time * 100) if total_time > 0 else 0
                st.markdown(f"**{label}:** {t:.1f}s ({pct:.0f}%)")
                st.progress(min(pct / 100, 1.0))

        # Per-agent cost breakdown
        if tracker.calls:
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
    st.markdown('<div class="main-header">Multi-Agent Debate System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Adversarial AI Debate powered by Groq (Free Tier)</div>',
                unsafe_allow_html=True)

    # Sidebar
    config = render_sidebar()

    # Load modules
    modules = load_pipeline_modules()
    if not modules:
        st.error("Failed to load pipeline. Check that all dependencies are installed and GROQ_API_KEY is set.")
        st.code("pip install -r requirements.txt", language="bash")
        return

    # Main content
    st.markdown("### Upload a PDF & Enter a Debate Topic")

    # File upload
    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload any PDF document as evidence source for the debate"
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

    # Debate topic input
    topic = st.text_input(
        "Debate Topic",
        placeholder='e.g., "Is Python better than Java for ML?" or "Should we use microservices?"',
        help="Enter a debatable topic related to the uploaded document"
    )

    # Example topics
    with st.expander("Example topics"):
        st.markdown("""
        - Is Python better than Java for machine learning?
        - Should companies adopt microservices architecture?
        - Is agile methodology better than waterfall?
        - Should AI replace human decision-making in healthcare?
        - Is remote work more productive than office work?
        """)

    # Run button
    if pdf_path and topic:
        st.markdown("---")

        col_run, col_info = st.columns([1, 3])
        with col_run:
            run_clicked = st.button("Start Debate", type="primary", use_container_width=True)
        with col_info:
            st.caption(
                f"Config: max_rounds={config['max_rounds']}, budget=${config['budget']:.2f} | "
                f"Rate limit: 4s delay between API calls"
            )

        if run_clicked:
            try:
                results = run_debate_with_ui(pdf_path, topic, config, modules)
                st.session_state["last_results"] = results
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

        elif "last_results" in st.session_state:
            display_results(st.session_state["last_results"])

    elif pdf_path and not topic:
        st.warning("Please enter a debate topic to start.")
    elif not pdf_path and topic:
        st.warning("Please upload a PDF document first.")

    # Footer
    st.markdown("---")
    st.caption(
        "Built with Streamlit | Powered by Groq (Free Tier) | "
        "Embeddings: all-MiniLM-L6-v2 (local) | Vector Search: FAISS"
    )


if __name__ == "__main__":
    main()
