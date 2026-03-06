# Multi-Agent Debate System - Complete Code Guide

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Setup & Prerequisites](#3-setup--prerequisites)
4. [File-by-File Code Walkthrough](#4-file-by-file-code-walkthrough)
   - [helpers.py - Shared Utilities](#41-helperspy---shared-utilities)
   - [project_debate_system.py - Complete Solution](#42-project_debate_systempy---complete-solution)
   - [debate_starter.py - Student Starter Code](#43-debate_starterpy---student-starter-code)
5. [Key Engineering Concepts](#5-key-engineering-concepts)
6. [Groq Free Tier Rate Limits](#6-groq-free-tier-rate-limits)
7. [How to Run](#7-how-to-run)
8. [Streamlit UI Guide](#8-streamlit-ui-guide)
9. [Improvement Ideas for Students](#9-improvement-ideas-for-students)

---

## 1. Project Overview

### What Does This Project Do?

This is a **Multi-Agent Adversarial Debate System** - a system that takes a PDF document and a debate topic, then orchestrates 7 specialized AI agents to conduct a structured debate with evidence, cross-examination, and judicial scoring:

| Agent | Role | Model Used | When Called |
|-------|------|------------|-------------|
| **Planner** | Sets up fair debate framework (positions, dimensions) | Llama 3.1 8B | Always (Round 1) |
| **Researcher FOR** | Gathers evidence supporting the FOR position | None (vector search only) | Always (Round 1) |
| **Researcher AGAINST** | Gathers evidence supporting the AGAINST position | None (vector search only) | Always (Round 1) |
| **Debater FOR** | Builds persuasive arguments with evidence | Llama 3.1 8B | Always (Round 1) |
| **Debater AGAINST** | Builds opposing arguments with evidence | Llama 3.1 8B | Always (Round 1) |
| **Judge** | Scores both sides on 5 criteria (0-10 each) | Llama 3.1 8B | Always (Round 1 + optional Round 2) |
| **Cross-Examiner FOR** | Challenges the AGAINST argument's weakest point | Llama 3.1 8B | Only if Round 1 is close |
| **Cross-Examiner AGAINST** | Challenges the FOR argument's weakest point | Llama 3.1 8B | Only if Round 1 is close |
| **Synthesizer** | Writes balanced analysis beyond "both sides have merit" | Llama 3.1 8B | Always (final) |

### What Makes This Different from Intelligence Hub?

| Aspect | Intelligence Hub | Debate System |
|--------|-----------------|---------------|
| **Goal** | Analyze one document comprehensively | Argue both sides of a topic using document evidence |
| **Agent interaction** | Independent (parallel, no communication) | Adversarial (agents read each other's outputs) |
| **Quality control** | Critic agent validates outputs | Judge scores + optional cross-examination round |
| **Adaptive behavior** | Retry if Critic score < 0.7 | Extra round if Judge margin is "narrow" or "razor-thin" |
| **Evidence use** | All agents share the same chunks | Each side has its own targeted evidence |

### Technologies Used

| Technology | Purpose | Cost |
|------------|---------|------|
| **Groq API** | LLM inference (Llama 3.1 8B) | Free tier available |
| **sentence-transformers** | Text embeddings (all-MiniLM-L6-v2) | Free (runs locally) |
| **FAISS** | Vector similarity search for evidence retrieval | Free (runs locally) |
| **PyMuPDF (fitz)** | PDF text extraction | Free |
| **python-dotenv** | API key management | Free |
| **Streamlit** | Interactive web UI (optional) | Free |

---

## 2. Architecture Diagram

### Full Pipeline Flow

```
                    ┌────────────────────────────────────────────┐
                    │  INPUTS: PDF Document + Debate Topic        │
                    └──────────────────┬─────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────┐
                    │  [1] Load & Chunk (PyMuPDF)                 │
                    │  300 words/chunk, 50 overlap                 │
                    └──────────────────┬─────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────┐
                    │  [2] Build FAISS Index                       │
                    │  Embed all chunks (local model, no API)      │
                    └──────────────────┬─────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────┐
                    │  [3] PLANNER Agent                           │
                    │  Analyzes topic + document context           │
                    │  Outputs: FOR position, AGAINST position,    │
                    │           debate dimensions                  │
                    │  Model: Llama 3.1 8B (call_llm_cheap)       │
                    └──────────────────┬─────────────────────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │                                           │
    ┌────────────▼────────────┐             ┌───────────────▼────────────┐
    │  [4a] RESEARCHER (FOR)   │             │  [4b] RESEARCHER (AGAINST)  │
    │  Vector search for       │  PARALLEL   │  Vector search for          │
    │  supporting evidence     │◄───────────►│  opposing evidence          │
    │  (No LLM - pure search)  │             │  (No LLM - pure search)     │
    └────────────┬────────────┘             └───────────────┬────────────┘
                 │                                           │
                 │          evidence_for (8 chunks)          │
                 │          evidence_against (8 chunks)      │
                 │                                           │
    ┌────────────▼────────────┐             ┌───────────────▼────────────┐
    │  [5a] DEBATER (FOR)      │             │  [5b] DEBATER (AGAINST)     │
    │  Builds 3 arguments      │  PARALLEL   │  Builds 3 arguments         │
    │  with evidence quotes    │◄───────────►│  with evidence quotes       │
    │  + preemptive rebuttal   │             │  + preemptive rebuttal      │
    │  Model: Llama 3.1 8B     │             │  Model: Llama 3.1 8B        │
    └────────────┬────────────┘             └───────────────┬────────────┘
                 │                                           │
                 └─────────────────┬─────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────┐
                    │  [6] JUDGE - Round 1                      │
                    │  Scores both sides (5 criteria, 0-10):    │
                    │    Evidence Quality | Logical Coherence    │
                    │    Completeness | Persuasiveness | Honesty │
                    │  Total: /50 each side                     │
                    │  Verdict: winner + margin                 │
                    │  Model: Llama 3.1 8B (call_llm_strong)   │
                    └──────────────┬──────────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │ Margin check:    │
                          │ "razor-thin" or  │
                          │ "narrow"?        │
                          └──┬───────────┬──┘
                         YES │           │ NO (decisive)
                             │           │
        ┌────────────────────▼──┐        │
        │  ROUND 2 (optional)   │        │
        │                       │        │
        │  ┌─────────────────┐  │        │
        │  │[7a] CROSS-EXAM  │  │        │
        │  │(FOR challenges  │  │        │
        │  │ AGAINST)        │  │        │
        │  │                 │  │        │
        │  │[7b] CROSS-EXAM  │  │        │
        │  │(AGAINST chall.  │  │        │
        │  │ FOR)  PARALLEL  │  │        │
        │  └────────┬────────┘  │        │
        │           │           │        │
        │  ┌────────▼────────┐  │        │
        │  │[7c] JUDGE R2    │  │        │
        │  │(with cross-exam │  │        │
        │  │ factored in)    │  │        │
        │  └────────┬────────┘  │        │
        │           │           │        │
        └───────────┼───────────┘        │
                    │                    │
                    └──────────┬─────────┘
                               │
                    ┌──────────▼──────────────────────────┐
                    │  [8] SYNTHESIZER                      │
                    │  Balanced analysis (150-200 words)     │
                    │  Common ground + key tension           │
                    │  Nuanced conclusion                   │
                    │  Model: Llama 3.1 8B (call_llm_strong)│
                    └──────────┬──────────────────────────┘
                               │
                    ┌──────────▼──────────────────────────┐
                    │  [9] REPORT COMPILER                  │
                    │  Assembles: Framework + Arguments +    │
                    │  Cross-Exam + Verdict + Synthesis     │
                    │  (Pure Python - no LLM needed)        │
                    └──────────┬──────────────────────────┘
                               │
                    ┌──────────▼──────────────────────────┐
                    │  [10] DIAGNOSTICS                     │
                    │  Agent execution log + Cost report    │
                    └──────────────────────────────────────┘
```

### Data Flow Between Agents

```
Planner ─── debate_plan ──────────────┬──────────────────────────────────┐
    │                                 │                                  │
    │  for_position, against_position │                                  │
    │  dimensions                     │                                  │
    ▼                                 ▼                                  ▼
Researcher FOR                   Researcher AGAINST               (used by Judge
    │                                 │                            & Synthesizer)
    │ evidence_for (8 chunks)         │ evidence_against (8 chunks)
    ▼                                 ▼
Debater FOR ──────────────┐    Debater AGAINST ──────────────┐
    │                     │         │                        │
    │ argument_for        │         │ argument_against       │
    ▼                     │         ▼                        │
    └────────┬────────────┘         └────────┬───────────────┘
             │                               │
             ▼                               ▼
         ┌───────────────────────────────────────┐
         │  JUDGE reads BOTH arguments            │
         │  Outputs: scores, winner, margin       │
         └───────────────┬───────────────────────┘
                         │
         ┌───────────────▼───────────────────────┐
         │  If margin is close:                   │
         │  Cross-Examiner FOR reads argument_    │
         │     against + evidence_for             │
         │  Cross-Examiner AGAINST reads argument_│
         │     for + evidence_against             │
         │  Judge R2 reads EVERYTHING             │
         └───────────────┬───────────────────────┘
                         │
         ┌───────────────▼───────────────────────┐
         │  SYNTHESIZER reads:                    │
         │  - Both arguments (thesis + points)    │
         │  - Judge verdict + reasoning           │
         │  - Strongest/weakest points per side   │
         └───────────────────────────────────────┘
```

### Agent Call Summary

```
ROUND 1 (always runs):                                    LLM Calls
  ├── Planner ─────────────────────── call_llm_cheap ────── 1
  ├── Researcher FOR ──────────────── (vector search) ──── 0
  ├── Researcher AGAINST ──────────── (vector search) ──── 0
  ├── Debater FOR ─────────────────── call_llm_strong ──── 1
  ├── Debater AGAINST ─────────────── call_llm_strong ──── 1
  └── Judge R1 ────────────────────── call_llm_strong ──── 1
                                                     Total: 4

ROUND 2 (only if margin is "narrow" or "razor-thin"):
  ├── Cross-Examiner FOR ──────────── call_llm_strong ──── 1
  ├── Cross-Examiner AGAINST ──────── call_llm_strong ──── 1
  └── Judge R2 ────────────────────── call_llm_strong ──── 1
                                                     Total: 3

SYNTHESIS (always runs):
  └── Synthesizer ─────────────────── call_llm_strong ──── 1
                                                     Total: 1

GRAND TOTAL: 5-8 LLM calls per debate
```

---

## 3. Setup & Prerequisites

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**What each package does:**

| Package | Import Name | Purpose |
|---------|-------------|---------|
| `groq` | `from groq import Groq` | API client to call Llama models on Groq's free tier |
| `sentence-transformers` | `from sentence_transformers import SentenceTransformer` | Converts text to 384-dimensional vectors for semantic evidence search |
| `pymupdf` | `import fitz` | Extracts text from PDF pages (note: import name differs!) |
| `faiss-cpu` | `import faiss` | Facebook's library for fast vector similarity search |
| `numpy` | `import numpy as np` | Numerical array operations |
| `pydantic` | `from pydantic import BaseModel` | Structured data validation |
| `python-dotenv` | `from dotenv import load_dotenv` | Loads API keys from `.env` file |
| `streamlit` | `import streamlit as st` | Interactive web UI (optional) |

### Step 2: Get API Key

```bash
cp .env.example .env
# Edit .env and add your Groq API key
# Get a free key at: https://console.groq.com/keys
```

### Step 3: Verify Setup

```bash
python helpers.py
# Should print "All tests passed. Ready for labs."
```

---

## 4. File-by-File Code Walkthrough

---

### 4.1 `helpers.py` - Shared Utilities

This is the **foundation library** shared with Intelligence Hub. It provides reusable building blocks for PDF processing, embeddings, LLM calls, and cost tracking. See the [Intelligence Hub Guide](../intelligence-hub/INTELLIGENCE_HUB_GUIDE.md) for a detailed walkthrough of helpers.py internals.

**Key settings for the Debate System:**

| Setting | Value | Why |
|---------|-------|-----|
| `RATE_LIMIT_DELAY` | 5 seconds | Prevents hitting Groq TPM limits |
| `call_llm_cheap` max_tokens | 500 | Planner JSON is small |
| `call_llm_strong` max_tokens | 1024 | Conserves free tier budget |
| `retries` | 1 | Fewer wasted calls on free tier |
| Model (all calls) | `llama-3.1-8b-instant` | 8b has 14.4K RPD + 500K TPD (generous) |

---

### 4.2 `project_debate_system.py` - Complete Solution

This is the **full implementation** that students build towards.

---

#### 4.2.1 Agent Prompts (Lines 35-110)

Each agent has a carefully crafted system prompt with explicit JSON output schema:

**Planner Prompt:**
```python
PLANNER_SYSTEM = """You are a Debate Planner. Set up a fair debate framework.
Return JSON:
{
    "topic_restated": "clear neutral statement of the debate",
    "for_position": "what the FOR side argues (1 sentence)",
    "against_position": "what the AGAINST side argues (1 sentence)",
    "dimensions": ["dimension1", "dimension2", "dimension3"],
    "context_from_document": "what the document says about this topic"
}"""
```

**Key prompt engineering principles:**
1. **Role assignment:** "You are a Debate Planner" / "You are a skilled Debater" / "You are an impartial Judge"
2. **Explicit rules:** Numbered rules prevent common mistakes (e.g., "Use ONLY evidence from chunks")
3. **JSON schema:** Exact output format ensures parseable responses
4. **Grounding:** "No training data" prevents hallucination
5. **Fairness:** Judge scores on 5 explicit criteria to prevent bias

**Debater Prompt - Special features:**
```python
DEBATER_SYSTEM_TEMPLATE = """You are a skilled Debater arguing {side}...
Rules:
- 3 clear arguments, each with: point, specific evidence quote, reasoning.
- Anticipate and preemptively counter the strongest opposing argument.
```
- Uses **template variables** (`{side}`, `{position}`) for both FOR and AGAINST sides
- Forces **evidence grounding** - arguments must include direct quotes from chunks
- Requires **preemptive rebuttal** - anticipating the other side's strongest point

**Judge Prompt - Scoring rubric:**
```python
JUDGE_SYSTEM = """Criteria (0-10 each):
1. Evidence Quality - claims backed by specific document evidence?
2. Logical Coherence - argument flows logically, no fallacies?
3. Completeness - addresses all debate dimensions?
4. Persuasiveness - how compelling overall?
5. Honesty - fair evidence representation or cherry-picking?
```
- 5 criteria x 10 points = 50 max per side
- Reports: winner, margin (decisive/narrow/razor-thin), reasoning, strongest/weakest points

---

#### 4.2.2 Agent Functions (Lines 117-309)

All agents follow a consistent pattern:

```python
def agent_name(state, tracker, side="for"):
    # 1. Extract inputs from shared state
    evidence = state.get(f"evidence_{side}", [])

    # 2. Call LLM with agent-specific prompt
    result = call_llm_strong(
        system=AGENT_SYSTEM_PROMPT,
        prompt=f"Topic: {state['topic']}\nEvidence:\n{evidence_text}",
        json_output=True
    )

    # 3. Track cost
    tracker.record(result, f"agent_{side}")

    # 4. Parse JSON and store in state
    parsed = parse_json(result["text"])
    state[f"output_{side}"] = parsed

    # 5. Log execution
    log_agent(state, f"agent_{side}", input_summary, output_summary, meta)

    return state
```

**Key concept - Consistent interface:** Every agent takes `(state, tracker)` or `(state, tracker, side)` and returns `state`. This enables parallel execution and pluggable agents.

**Researcher agent is special** - it uses NO LLM calls:
```python
def researcher(state, tracker, side="for"):
    """Search for evidence supporting one side. No LLM needed - pure vector search."""
    queries = [
        f"evidence supporting {plan['for_position']}",
        f"benefits advantages {topic}",
    ]
    for q in queries:
        evidence.extend(search(state["_index"], state["_all_chunks"], q, k=3))
```
This is a deliberate design choice: evidence gathering is pure vector search, keeping LLM calls for reasoning tasks where they add the most value.

**Cross-examiner reads the opponent's output:**
```python
def cross_examiner(state, tracker, my_side="for"):
    opposing_side = "against" if my_side == "for" else "for"
    opposing_arg = state.get(f"argument_{opposing_side}", {})  # reads opponent!
```
This is the **adversarial** aspect - Cross-examiners are the only agents that directly read another agent's output to challenge it.

---

#### 4.2.3 Parallel Execution with Error Boundaries (Lines 416-441)

```python
def run_parallel(fns_with_args, state, tracker):
    with ThreadPoolExecutor(max_workers=len(fns_with_args)) as executor:
        futures = {}
        for fn, side in fns_with_args:
            s = {**state}  # COPY state for each agent
            futures[executor.submit(fn, s, tracker, side)] = (fn.__name__, side)

        for future in as_completed(futures):
            name, side = futures[future]
            try:
                result_state = future.result()
                results[side] = result_state
            except Exception as e:
                state["errors"].append({"agent": f"{name}_{side}", "error": str(e)})
```

**Key differences from Intelligence Hub's parallel execution:**

| Aspect | Intelligence Hub | Debate System |
|--------|-----------------|---------------|
| **What runs in parallel** | 4 independent agents | 2 adversarial pairs (FOR/AGAINST) |
| **State merging** | Each agent writes to unique keys | Merge by side: `evidence_for`, `argument_for`, etc. |
| **Used at** | One stage | Three stages: Research, Debate, Cross-examination |

**Why state copying matters:** `s = {**state}` creates a shallow copy. Without this, both threads would write to the same dict simultaneously, causing race conditions. After parallel execution, results are merged back by side key.

---

#### 4.2.4 Conditional Round 2 (Lines 498-515)

```python
margin = state.get("judgment", {}).get("margin", "decisive")
if max_rounds > 1 and margin in ("razor-thin", "narrow") and tracker.remaining() > 0.15:
    # Cross-examination + re-judge
    state = run_parallel([(cross_examiner, "for"), (cross_examiner, "against")], ...)
    state = judge(state, tracker, round_num=2)
    state["rounds_played"] = 2
else:
    if margin not in ("razor-thin", "narrow"):
        print(f"Skipping cross-examination: result was {margin}")
```

**Three conditions must ALL be true for Round 2:**
1. `max_rounds > 1` - User allows multiple rounds
2. `margin in ("razor-thin", "narrow")` - Initial verdict was close
3. `tracker.remaining() > 0.15` - Enough budget for 3 more LLM calls

**Key concept - Adaptive behavior:** The pipeline doesn't blindly run all stages. It makes a runtime decision based on the Judge's output. This is more efficient and realistic than always running cross-examination.

---

#### 4.2.5 Report Formatting (Lines 316-409)

```python
def format_report(state):
    r = []
    r.append("  MULTI-AGENT DEBATE REPORT")
    r.append(f"  Topic:  {state.get('topic', '?')}")
    r.append(f"  FOR:     {plan.get('for_position', '?')}")
    # ... assembles all sections ...
```

**Report sections:**
1. Debate Framework (topic, positions, dimensions)
2. FOR Argument (thesis, 3 arguments with evidence, rebuttal)
3. AGAINST Argument (thesis, 3 arguments with evidence, rebuttal)
4. Cross-Examination (if played)
5. Judge's Verdict (score table, winner, reasoning)
6. Balanced Synthesis (analysis, common ground, tension, conclusion)

Pure Python - no LLM needed. Deterministic formatting.

---

#### 4.2.6 Main Pipeline (Lines 448-529)

```python
def run_debate(pdf_path, topic, budget=0.80, max_rounds=2):
    tracker = CostTracker(budget=budget)

    # [1] Load + index
    chunks = load_and_chunk(pdf_path)
    index = build_index(chunks)

    # [2] Plan
    state = debate_planner(state, tracker)

    # [3] Research (parallel FOR + AGAINST)
    state = run_parallel([(researcher, "for"), (researcher, "against")], ...)

    # [4] Debate (parallel FOR + AGAINST)
    state = run_parallel([(debater, "for"), (debater, "against")], ...)

    # [5] Judge Round 1
    state = judge(state, tracker, round_num=1)

    # [6] Conditional: Cross-examination + Judge Round 2
    if margin is close and budget allows:
        state = run_parallel([(cross_examiner, "for"), (cross_examiner, "against")], ...)
        state = judge(state, tracker, round_num=2)

    # [7] Synthesis
    state = synthesizer(state, tracker)

    # Report + diagnostics
    report = format_report(state)
    tracker.report()
```

**Key concept - Budget checks between stages:** `tracker.check_budget()` is called after each stage. If a student sets `--budget 0.10`, the pipeline stops gracefully instead of making API calls it can't afford.

---

### 4.3 `debate_starter.py` - Student Starter Code

This file provides the agent prompts but leaves the **pipeline engineering** for students to build:

| Milestone | Time | What You Build |
|-----------|------|----------------|
| **1** | 45 min | Wire sequential pipeline (Planner -> Research -> Debate -> Judge) |
| **2** | 45 min | Add parallel execution for Research and Debate stages |
| **3** | 30 min | Add cost tracking and budget enforcement between stages |
| **4** | 30 min | Implement conditional Round 2 (cross-examination logic) |
| **5** | 30 min | Add Synthesizer + Report compiler |

---

## 5. Key Engineering Concepts

### 5.1 Adversarial Multi-Agent Architecture

**Pattern:** Agents don't just work independently - they argue against each other.

```
Intelligence Hub: Agents work in PARALLEL (no interaction)
   Summarizer ──┐
   Fact Extr. ──┤── Critic validates all
   Quiz Gen.  ──┤
   Gap Anal.  ──┘

Debate System: Agents work ADVERSARIALLY (read each other's outputs)
   Debater FOR ──────► Judge ◄────── Debater AGAINST
        ▲                                   ▲
        │                                   │
   Cross-Exam FOR ◄── reads ──► Cross-Exam AGAINST
```

### 5.2 Evidence-Grounded Argumentation

**Pattern:** Arguments must be backed by specific quotes from the document.

```
BAD:  "Python is better because it's popular"  (opinion, no evidence)
GOOD: "Python is better because, as stated in paragraph 3,
       'Python's scikit-learn library provides 200+ algorithms'"  (grounded)
```

The debater prompts enforce this: "Use ONLY evidence from the provided chunks. No training data."

### 5.3 Conditional Pipeline Branching

**Pattern:** The pipeline makes runtime decisions based on intermediate results.

```python
# Not this (always runs everything):
research() -> debate() -> cross_examine() -> judge() -> synthesize()

# But this (conditional branching):
research() -> debate() -> judge()
                            │
                     ┌──────┴──────┐
                  decisive?      close?
                     │              │
                  synthesize()   cross_examine() -> judge() -> synthesize()
```

### 5.4 Parallel Adversarial Pairs

**Pattern:** Both sides of the debate run simultaneously since they're independent.

```python
# Research stage: FOR and AGAINST search independently
run_parallel([(researcher, "for"), (researcher, "against")])

# Debate stage: both debaters build arguments independently
run_parallel([(debater, "for"), (debater, "against")])

# Cross-exam: both sides challenge the other simultaneously
run_parallel([(cross_examiner, "for"), (cross_examiner, "against")])
```

Each parallel pair gets its own state copy. Results are merged by side key (`evidence_for`, `evidence_against`, etc.).

### 5.5 Structured Scoring Rubric

**Pattern:** The Judge doesn't just say "FOR wins." It scores on 5 explicit criteria:

```
Criterion            FOR    AGAINST
Evidence Quality     8/10   7/10
Logical Coherence    7/10   8/10
Completeness         8/10   7/10
Persuasiveness       7/10   8/10
Honesty              9/10   8/10
─────────────────────────────────
TOTAL                39/50  38/50
Winner: FOR (narrow)
```

This rubric makes the judgment transparent, reproducible, and debuggable.

---

## 6. Groq Free Tier Rate Limits

### Understanding the Limits

| Model | RPM | RPD | TPM | TPD |
|-------|-----|-----|-----|-----|
| **llama-3.1-8b-instant** (used for all calls) | 30 | 14.4K | **6K** | **500K** |
| llama-3.3-70b-versatile (not used - too limited) | 30 | 1K | 12K | 100K |

**Why we use 8b only:** The 70b model has only 1K requests/day and 100K tokens/day. Students need to run multiple projects (Intelligence Hub + Debate System). Using 8b gives 14x more requests and 5x more tokens per day.

### Token Budget Per Debate Run

```
Stage                    LLM Calls    Est. Tokens (in+out)
─────────────────────────────────────────────────────────
Planner (8B)                  1          ~1,500
Debater FOR (8B)              1          ~3,000
Debater AGAINST (8B)          1          ~3,000
Judge R1 (8B)                 1          ~3,500
[Cross-Exam FOR] (8B)         1          ~2,500  (optional)
[Cross-Exam AGAINST] (8B)     1          ~2,500  (optional)
[Judge R2] (8B)               1          ~3,500  (optional)
Synthesizer (8B)              1          ~2,000
─────────────────────────────────────────────────────────
Round 1 only:           5 calls         ~13,000 tokens
With Round 2:           8 calls         ~21,500 tokens

Daily budget (8B):                      500,000 tokens
Debates per day (R1 only):              ~38
Debates per day (with R2):              ~23
```

Students can comfortably run 20+ debates per day and still have budget for Intelligence Hub.

### How Rate Limiting Works

```python
RATE_LIMIT_DELAY = 5  # seconds between calls

# Thread-safe rate limiting (handles parallel agents):
with _rate_limit_lock:
    elapsed = time.time() - _last_llm_call_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_llm_call_time = time.time()
```

When 2 debaters run in parallel, the rate limiter staggers their API calls 5 seconds apart.

---

## 7. How to Run

### CLI Usage

```bash
# Basic debate
python project_debate_system.py sample_docs/cloud_computing.pdf "Is cloud better than on-premise?"

# With budget and round limits
python project_debate_system.py sample_docs/neural_networks.pdf "Are CNNs better than RNNs?" --budget 0.30 --rounds 2

# All options
python project_debate_system.py your_doc.pdf "Your topic" --budget 0.80 --rounds 3
```

### Streamlit UI

```bash
pip install streamlit
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

### Starter Code (Learning Mode)

```bash
python debate_starter.py sample_docs/cloud_computing.pdf "Is cloud computing secure?"
```

---

## 8. Streamlit UI Guide

### File: `streamlit_app.py`

The Streamlit app imports from `helpers.py` and `project_debate_system.py` to provide a web-based interface.

```
streamlit_app.py
    ├── imports from helpers.py (load_and_chunk, build_index, etc.)
    ├── imports from project_debate_system.py (all agent functions)
    └── provides web UI with:
        ├── PDF upload + topic input
        ├── Pipeline configuration sidebar
        ├── Real-time progress bar (8 stages)
        ├── 8-tab results display
        └── Download button for reports
```

### UI Tabs

| Tab | What It Shows |
|-----|---------------|
| **Debate Framework** | Topic, FOR/AGAINST positions, dimensions, document context |
| **FOR Argument** | Thesis, 3 arguments with evidence, rebuttal, closing, evidence chunks |
| **AGAINST Argument** | Thesis, 3 arguments with evidence, rebuttal, closing, evidence chunks |
| **Cross-Examination** | Weakest points identified, challenges, unanswerable questions |
| **Judge's Verdict** | Score breakdown table (5 criteria), winner, strongest/weakest points |
| **Balanced Synthesis** | Analysis, common ground, key tension, nuanced conclusion |
| **Full Report** | Complete text report with download button |
| **Cost & Timing** | Token usage per agent, timing breakdown, budget status |

### Sidebar Features

- **API Key status** - green/yellow indicator, manual key entry
- **Debate config** - max rounds slider (1-3), budget slider ($0.05-$1.00)
- **Rate limit info** - current Groq free tier limits
- **Architecture diagram** - pipeline flow reference

---

## 9. Improvement Ideas for Students

### Beginner Level
1. **Add a Moderator agent** - Ensures debate stays on topic and dimensions are covered
2. **Better evidence display** - Show which chunks each debater used, with highlighted quotes
3. **Support more file types** - Add DOCX, TXT, HTML alongside PDF
4. **Add example topics** - Pre-loaded topic suggestions based on the uploaded document

### Intermediate Level
5. **Multi-round debate** - Allow 3+ rounds with escalating cross-examination
6. **Audience voting** - Add a simulated audience panel (3 LLM calls with different personas)
7. **Fact-checking agent** - Verify evidence quotes actually exist in the source document
8. **Debate transcript** - Generate a natural-language debate transcript (not just JSON)
9. **Comparison mode** - Run the same topic on two different documents, compare verdicts

### Advanced Level
10. **Async pipeline** - Replace ThreadPoolExecutor with asyncio for true async I/O
11. **Agent memory** - Debaters remember arguments from previous rounds
12. **Dynamic judge criteria** - Let the Planner define custom scoring criteria per topic
13. **Multi-document debate** - Each side uses a different source document
14. **Real-time streaming** - Stream debater arguments as they're generated

### Research Level
15. **Constitutional AI debate** - Agents debate ethical implications with safety constraints
16. **Debate tournament** - Multiple topics, elimination rounds, Elo ratings for agents
17. **Self-improving prompts** - Use Judge feedback to automatically improve Debater prompts
18. **Human-in-the-loop** - Allow a human to intervene as one of the debaters
19. **Benchmark suite** - Compare debate quality across different LLM models
20. **Knowledge graph** - Build argument dependency graphs and visualize logical chains

---

*This guide was created to help students understand the Multi-Agent Debate System codebase. Each section explains not just WHAT the code does, but WHY it's designed that way and HOW students can improve it.*
