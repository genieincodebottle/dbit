"""
Multi-Agent Document Intelligence Hub - STARTER CODE
=====================================================

WHAT'S GIVEN: Agent prompts and individual agent functions (they work standalone).
YOUR JOB:     Build the ENGINEERING that makes them work together.

Milestone 1 (45 min): Wire the pipeline - call agents in correct order, design state flow
Milestone 2 (45 min): Implement parallel execution with ThreadPoolExecutor + error boundaries
Milestone 3 (30 min): Add cost tracking (CostTracker) + budget enforcement
Milestone 4 (30 min): Build evaluation harness - test with 5 questions, measure quality
Milestone 5 (30 min): Adaptive retry - Critic feedback loops back to Planner on re-run

TESTING APPROACH:
  After each milestone, run: python hub_starter.py your_document.pdf
  The pipeline should progressively improve.

Usage:
  python hub_starter.py sample_docs/your_document.pdf
  python hub_starter.py your_doc.pdf --budget 0.30 --eval
"""

from helpers import (
    load_and_chunk, build_index, search,
    call_llm_cheap, call_llm_strong,
    parse_json, init_state, log_agent, print_log,
    CostTracker, SemanticCache, EvalHarness
)
# MILESTONE 2: You'll need this for parallel execution
# from concurrent.futures import ThreadPoolExecutor, as_completed
import json, sys, time, argparse, traceback


# ============================================================
# AGENT FUNCTIONS (GIVEN - these work individually)
# ============================================================
# These are complete. Your job is to wire them together into a pipeline.
# Each agent takes (state, tracker) and returns modified state.

PLANNER_SYSTEM = """You are a Document Planner agent.
Analyze the document and identify structure, themes, and complexity.
Return JSON:
{
    "document_type": "textbook chapter | research paper | policy doc | report | other",
    "main_topic": "one clear sentence",
    "key_themes": ["theme1", "theme2", "theme3", "theme4", "theme5"],
    "target_audience": "who this is written for",
    "estimated_complexity": "beginner | intermediate | advanced"
}"""

SUMMARIZER_SYSTEM = """You are a Summarizer agent. Write a concise executive summary.
Rules: Max 5 sentences. Every statement from chunks only. No training data.
Return JSON: {"summary": "3-5 sentences", "key_points": ["point1", "point2", "point3"]}"""

FACT_EXTRACTOR_SYSTEM = """You are a Fact Extractor agent. Extract 6-10 key facts.
Classify each as: fact | claim | formula | statistic. Rate: high | medium | low.
Return JSON: {"facts": [{"fact": "...", "type": "...", "importance": "..."}], "total_extracted": N}"""

QUIZ_SYSTEM = """You are a Quiz Generator agent. Create 5 MCQs testing UNDERSTANDING, not recall.
4 options (A-D), 1 correct. Mix: 2 easy, 2 medium, 1 hard. Include explanation.
Return JSON: {"questions": [{"question": "...", "options": {"A":"..","B":"..","C":"..","D":".."}, "correct": "B", "explanation": "...", "difficulty": "easy|medium|hard"}]}"""

GAP_ANALYZER_SYSTEM = """You are a Gap Analyzer agent. Find 3-5 missing or underexplained topics.
Be specific. Rate: critical | moderate | minor.
Return JSON: {"gaps": [{"topic": "...", "why_important": "...", "severity": "..."}], "coverage_score": 0.75, "recommendation": "one sentence"}"""

CRITIC_SYSTEM = """You are a Critic agent (quality gate). Verify all outputs against source chunks.
Be STRICT. Score 0 for any unsupported claim.
Return JSON:
{"overall_score": 0.85, "scores": {"summary": {"score": 0.9, "issues": []}, "facts": {"score": 0.8, "issues": []}, "quiz": {"score": 0.85, "issues": []}, "gaps": {"score": 0.9, "issues": []}}, "verdict": "pass|fail", "critical_issues": [], "improvement_hints": ["what to fix on retry"]}"""


def planner(state, tracker):
    """Analyzes document structure. Works standalone."""
    sample_text = "\n---\n".join([c["text"] for c in state["chunks"][:5]])  # reduced from 8 for free tier TPM

    # Adaptive: on retry, include critic feedback
    retry_hint = ""
    if state.get("_critic_feedback"):
        retry_hint = f"\n\nPrevious issues: {state['_critic_feedback']}. Adjust themes accordingly."

    result = call_llm_cheap(system=PLANNER_SYSTEM,
                            prompt=f"Analyze:\n\n{sample_text}{retry_hint}",
                            json_output=True)
    tracker.record(result, "planner")

    parsed = parse_json(result["text"])
    state["plan"] = parsed
    log_agent(state, "planner", {"chunks": 5}, parsed,
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Planner] Topic: {parsed.get('main_topic', '?')}")
    return state


def summarizer(state, tracker):
    """Writes executive summary. Works standalone."""
    chunks_text = "\n---\n".join([c["text"] for c in state["chunks"][:6]])  # reduced from 15 for free tier
    plan = state.get("plan", {})
    result = call_llm_strong(
        system=SUMMARIZER_SYSTEM,
        prompt=f"Topic: {plan.get('main_topic', '?')}\nThemes: {', '.join(plan.get('key_themes', []))}\n\nChunks:\n{chunks_text}",
        json_output=True)
    tracker.record(result, "summarizer")
    state["summary"] = parse_json(result["text"])
    log_agent(state, "summarizer", {"chunks": 6}, state["summary"].get("summary", "")[:80],
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Summarizer] Done")
    return state


def fact_extractor(state, tracker):
    """Extracts facts. Works standalone."""
    chunks_text = "\n---\n".join([c["text"] for c in state["chunks"][:6]])  # reduced from 15 for free tier
    result = call_llm_strong(system=FACT_EXTRACTOR_SYSTEM,
                             prompt=f"Extract facts:\n\n{chunks_text}", json_output=True)
    tracker.record(result, "fact_extractor")
    state["facts"] = parse_json(result["text"])
    log_agent(state, "fact_extractor", {"chunks": 6},
              f"{len(state['facts'].get('facts', []))} facts",
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Fact Extractor] {len(state['facts'].get('facts', []))} facts")
    return state


def quiz_generator(state, tracker):
    """Creates MCQs. Works standalone."""
    chunks_text = "\n---\n".join([c["text"] for c in state["chunks"][:6]])  # reduced from 15 for free tier
    plan = state.get("plan", {})
    result = call_llm_strong(
        system=QUIZ_SYSTEM,
        prompt=f"Topic: {plan.get('main_topic', '?')}\nSource:\n{chunks_text}",
        json_output=True)
    tracker.record(result, "quiz_generator")
    state["quiz"] = parse_json(result["text"])
    log_agent(state, "quiz_generator", {"chunks": 6},
              f"{len(state['quiz'].get('questions', []))} questions",
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Quiz Gen] {len(state['quiz'].get('questions', []))} questions")
    return state


def gap_analyzer(state, tracker):
    """Finds gaps. Works standalone."""
    chunks_text = "\n---\n".join([c["text"] for c in state["chunks"][:6]])  # reduced from 15 for free tier
    plan = state.get("plan", {})
    result = call_llm_strong(
        system=GAP_ANALYZER_SYSTEM,
        prompt=f"Type: {plan.get('document_type', '?')}\nTopic: {plan.get('main_topic', '?')}\n\n{chunks_text}",
        json_output=True)
    tracker.record(result, "gap_analyzer")
    state["gaps"] = parse_json(result["text"])
    log_agent(state, "gap_analyzer", {"chunks": 6},
              f"{len(state['gaps'].get('gaps', []))} gaps",
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Gap Analyzer] {len(state['gaps'].get('gaps', []))} gaps")
    return state


def critic(state, tracker):
    """Quality gate. Works standalone."""
    chunks_text = "\n---\n".join([c["text"] for c in state["chunks"][:5]])  # reduced from 10 for free tier
    outputs = {
        "summary": state.get("summary", {}).get("summary", "MISSING"),
        "facts_sample": [f["fact"] for f in state.get("facts", {}).get("facts", [])[:3]],
        "quiz_sample": state.get("quiz", {}).get("questions", [{}])[0].get("question", "N/A") if state.get("quiz", {}).get("questions") else "N/A",
        "gaps_count": len(state.get("gaps", {}).get("gaps", []))
    }
    result = call_llm_strong(
        system=CRITIC_SYSTEM,
        prompt=f"Source chunks:\n{chunks_text}\n\nValidate:\n{json.dumps(outputs, indent=2)}",
        json_output=True)
    tracker.record(result, "critic")
    parsed = parse_json(result["text"])
    state["critic"] = parsed
    state["critic_score"] = parsed.get("overall_score", 0)
    state["_critic_feedback"] = "; ".join(parsed.get("improvement_hints", []))
    log_agent(state, "critic", {"outputs": 4},
              f"Score={state['critic_score']}, {parsed.get('verdict', '?')}",
              {"tokens": result["tokens"], "latency_ms": result["latency_ms"]})
    print(f"  [Critic] {state['critic_score']}/1.0 - {parsed.get('verdict', '?').upper()}")
    return state


# ============================================================
# MILESTONE 1: WIRE THE SEQUENTIAL PIPELINE
# ============================================================
# YOUR TASK:
#   1. Load the PDF and build the FAISS index
#   2. Select diverse chunks (search for different aspects)
#   3. Call agents in the correct order: planner -> analysis agents -> critic
#   4. Print results
#
# CONSTRAINT: Run agents SEQUENTIALLY first (parallel comes in Milestone 2)
# TEST: python hub_starter.py your_doc.pdf
#       You should see output from each agent in order.

def run_pipeline(pdf_path, budget=0.10, max_retries=1):
    tracker = CostTracker(budget=budget)

    print(f"\n{'=' * 60}")
    print(f"  DOCUMENT INTELLIGENCE HUB")
    print(f"{'=' * 60}")

    # --- YOUR CODE: Step 1 - Load and index the PDF ---
    # HINT: chunks = load_and_chunk(pdf_path)
    # HINT: index = build_index(chunks)
    print("\n[1] Loading document...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 2 - Select diverse representative chunks ---
    # HINT: Search for different aspects of the document to get diversity.
    #       Don't just take the first 20 chunks - search for "overview",
    #       "conclusion", "methodology", etc. and combine results.
    # HINT: Deduplicate by chunk index (same chunk may match multiple queries).
    # HINT: Store in state["chunks"] (list of {text, score, index} dicts)
    print("\n[2] Selecting chunks...")
    state = init_state()
    # YOUR CODE HERE


    # --- YOUR CODE: Step 3 - Run Planner ---
    print("\n[3] Running Planner...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 4 - Run the 4 analysis agents ---
    # For Milestone 1: run them SEQUENTIALLY (one after another).
    # Milestone 2 will make this parallel.
    print("\n[4] Running analysis agents...")
    # YOUR CODE HERE (call summarizer, fact_extractor, quiz_generator, gap_analyzer)


    # --- YOUR CODE: Step 5 - Run Critic (quality gate) ---
    print("\n[5] Running Critic...")
    # YOUR CODE HERE


    # --- YOUR CODE: Step 6 - Print results ---
    # HINT: state has keys: plan, summary, facts, quiz, gaps, critic, critic_score
    # Print at least: critic_score, summary text, number of facts, number of questions
    print("\n[6] Results:")
    # YOUR CODE HERE


    print_log(state)
    tracker.report()
    return state


# ============================================================
# MILESTONE 2: PARALLEL EXECUTION + ERROR BOUNDARIES
# ============================================================
# YOUR TASK:
#   Replace the sequential agent calls in Step 4 with ThreadPoolExecutor.
#   Run summarizer, fact_extractor, quiz_generator, gap_analyzer in PARALLEL.
#
# REQUIREMENTS:
#   1. Import ThreadPoolExecutor and as_completed at the top of the file
#   2. Each agent gets its OWN COPY of state (to avoid race conditions)
#   3. Wrap each future.result() in try/except - one failure shouldn't crash all
#   4. Merge results back into main state after all complete
#   5. Append each agent's logs to the main state["log"]
#
# HINT: See Python docs for concurrent.futures.ThreadPoolExecutor
# HINT: To copy state: agent_state = {**state}
# HINT: To merge: state["summary"] = result_state["summary"]
#
# TEST: Run the pipeline - it should be ~3x faster than Milestone 1.
#       Time it: add start = time.time() before and elapsed = time.time() - start after.
#       Compare sequential vs parallel timing.

def run_parallel_agents(state, tracker):
    """
    YOUR CODE: Run 4 analysis agents in parallel.
    Return the modified state with all agent outputs merged.
    """
    # YOUR CODE HERE
    pass


# ============================================================
# MILESTONE 3: COST TRACKING + BUDGET ENFORCEMENT
# ============================================================
# YOUR TASK:
#   1. The CostTracker is already passed to each agent (tracker param).
#      Each agent already calls tracker.record(). That part is done.
#   2. Add tracker.check_budget() BETWEEN pipeline stages.
#      If budget is exceeded, it raises RuntimeError. Catch it and stop gracefully.
#   3. Print tracker.report() at the end of the pipeline.
#   4. Try running with --budget 0.05 to see it hit the budget limit.
#
# ENGINEERING QUESTION: Where should you place budget checks?
#   - After every agent? (safe but verbose)
#   - After each pipeline stage? (practical)
#   - Only at the end? (too late - already spent the money)


# ============================================================
# MILESTONE 4: EVALUATION HARNESS
# ============================================================
# YOUR TASK:
#   Build an evaluation mode that tests your pipeline on 5 questions.
#
# REQUIREMENTS:
#   1. Create an EvalHarness with 5 test cases for your specific document.
#      Each test case needs: question, expected_keywords, difficulty.
#   2. Write a pipeline_for_eval(question) function that:
#      - Loads the document (or reuses cached index)
#      - Searches for relevant chunks
#      - Runs planner -> analysis -> critic
#      - Returns {"answer": state["summary"]["summary"], "critic_score": state["critic_score"]}
#   3. Run harness.run(pipeline_for_eval) and harness.report()
#
# HINT: The EvalHarness checks if expected_keywords appear in the answer
#       and combines with the critic_score for a composite score.
# HINT: Reuse the FAISS index across test cases (don't rebuild every time).
#
# TEST: python hub_starter.py your_doc.pdf --eval
#       Target: >60% pass rate on your 5 test questions.

def run_evaluation(pdf_path):
    """
    YOUR CODE: Build and run evaluation harness.
    """
    # YOUR CODE HERE
    pass


# ============================================================
# MILESTONE 5: ADAPTIVE RETRY
# ============================================================
# YOUR TASK:
#   When the Critic scores < 0.7, don't just re-run blindly.
#   Feed the Critic's feedback BACK to the Planner so it adapts.
#
# REQUIREMENTS:
#   1. Build a retry loop: while retry < max_retries and score < 0.7
#   2. On retry, state["_critic_feedback"] already contains the Critic's hints
#      (the planner() function already reads this - check its code)
#   3. Re-run: planner (with feedback) -> parallel agents -> critic
#   4. Track retry_count in state
#   5. Budget check before each retry
#
# TEST: Find a question that scores < 0.7 on first attempt.
#       Does the score improve on retry? Check the log to see if Planner adapted.
#
# ENGINEERING QUESTION: When should you NOT retry?
#   - Budget too low for another full cycle?
#   - Score is 0.0 (fundamentally broken, retrying won't help)?
#   - Already retried max_retries times?


# ============================================================
# CLI ENTRY POINT (GIVEN)
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Intelligence Hub")
    parser.add_argument("pdf", help="Path to PDF document")
    parser.add_argument("--budget", type=float, default=0.10, help="Budget in USD")
    parser.add_argument("--retries", type=int, default=1, help="Max retries (reduced for free tier)")
    parser.add_argument("--eval", action="store_true", help="Run evaluation harness")

    args = parser.parse_args()

    state = run_pipeline(args.pdf, budget=args.budget, max_retries=args.retries)

    if args.eval:
        print("\n\nRUNNING EVALUATION...")
        run_evaluation(args.pdf)
