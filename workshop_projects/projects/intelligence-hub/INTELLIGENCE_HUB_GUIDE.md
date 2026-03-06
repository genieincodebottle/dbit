# Multi-Agent Document Intelligence Hub - Complete Code Guide

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Setup & Prerequisites](#3-setup--prerequisites)
4. [File-by-File Code Walkthrough](#4-file-by-file-code-walkthrough)
   - [helpers.py - Shared Utilities](#41-helperspy---shared-utilities)
   - [project_intelligence_hub.py - Complete Solution](#42-project_intelligence_hubpy---complete-solution)
   - [hub_starter.py - Student Starter Code](#43-hub_starterpy---student-starter-code)
5. [Key Engineering Concepts](#5-key-engineering-concepts)
6. [Groq Free Tier Rate Limits](#6-groq-free-tier-rate-limits)
7. [How to Run](#7-how-to-run)
8. [Streamlit UI Guide](#8-streamlit-ui-guide)
9. [Improvement Ideas for Students](#9-improvement-ideas-for-students)

---

## 1. Project Overview

### What Does This Project Do?

This is a **Multi-Agent Document Intelligence Hub** - a system that takes any PDF document and produces a comprehensive intelligence report by coordinating 7 specialized AI agents:

| Agent | Role | Model Used |
|-------|------|------------|
| **Planner** | Analyzes document structure, themes, complexity | Llama 3.1 8B (cheap/fast) |
| **Summarizer** | Writes executive summary with key points | Llama 3.1 8B (strong) |
| **Fact Extractor** | Extracts and classifies important facts | Llama 3.1 8B (strong) |
| **Quiz Generator** | Creates MCQs that test understanding | Llama 3.1 8B (strong) |
| **Gap Analyzer** | Identifies missing or underexplained topics | Llama 3.1 8B (strong) |
| **Critic** | Quality gate - validates all outputs against source | Llama 3.1 8B (strong) |
| **Report Compiler** | Assembles final report (no LLM needed) | None (pure Python) |

### Why Multiple Agents?

Instead of one monolithic prompt, this project uses the **divide-and-conquer** principle:
- Each agent has a **single, clear responsibility** (Single Responsibility Principle)
- Agents can run **in parallel** (4 analysis agents run simultaneously)
- Failures are **isolated** - one agent crashing doesn't kill the whole pipeline
- Quality is **validated** by an independent Critic agent

### Technologies Used

| Technology | Purpose | Cost |
|------------|---------|------|
| **Groq API** | LLM inference (Llama models) | Free tier available |
| **sentence-transformers** | Text embeddings (all-MiniLM-L6-v2) | Free (runs locally) |
| **FAISS** | Vector similarity search | Free (runs locally) |
| **PyMuPDF (fitz)** | PDF text extraction | Free |
| **python-dotenv** | API key management | Free |

---

## 2. Architecture Diagram

```
                    ┌─────────────────────────────┐
                    │       PDF Document           │
                    └──────────┬──────────────────-┘
                               │
                    ┌──────────▼──────────────────-┐
                    │  [1] Load & Chunk (PyMuPDF)   │
                    │  300 words/chunk, 50 overlap   │
                    └──────────┬──────────────────-─┘
                               │
                    ┌──────────▼──────────────────-─┐
                    │  [2] Build FAISS Index          │
                    │  Embed all chunks (local model) │
                    └──────────┬──────────────────-─-┘
                               │
                    ┌──────────▼──────────────────-─┐
                    │  [3] Select Diverse Chunks      │
                    │  Search 5 aspects → deduplicate  │
                    └──────────┬──────────────────-─-┘
                               │
                    ┌──────────▼──────────────────-─-┐
                    │  [4] PLANNER Agent               │
                    │  Identifies type, topic, themes  │
                    │  Model: Llama 3.1 8B (cheap)     │
              ┌─────└──────────┬──────────────────-─-┘
              │                │
              │   ┌────────────┼────────────┬────────────┐
              │   │            │            │            │
              │   ▼            ▼            ▼            ▼
              │ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
              │ │Summa-  │ │Fact    │ │Quiz    │ │Gap     │
              │ │rizer   │ │Extract.│ │Generat.│ │Analyz. │
              │ │        │ │        │ │        │ │        │
              │ │[5] PARALLEL EXECUTION (ThreadPoolExecutor) │
              │ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
              │     │          │          │          │
              │     └──────────┴──────┬───┴──────────┘
              │                       │
              │          ┌────────────▼───────────────┐
              │          │  [6] CRITIC Agent           │
              │          │  Validates all outputs      │
              │          │  Score >= 0.7 → PASS        │
              │          │  Score <  0.7 → RETRY       │
              │          └────────────┬───────────────┘
              │                       │
              │              ┌────────▼────────┐
              │              │  Score >= 0.7?   │
              │              └──┬───────────┬──┘
              │             YES │           │ NO (retry)
              │                 │           │
              │                 │    ┌──────▼──────────┐
              │                 │    │ Feed critic      │
              └─────────────────┼────│ feedback back    │
               (adaptive retry) │    │ to Planner       │
                                │    └─────────────────┘
                       ┌────────▼────────────┐
                       │  [7] Report Compiler  │
                       │  Assembles final doc  │
                       └────────┬────────────-┘
                                │
                       ┌────────▼────────────┐
                       │  [8] Diagnostics     │
                       │  Cost + Execution Log │
                       └──────────────────────┘
```

### Adaptive Retry Flow

```
Attempt 1: Planner → [4 agents in parallel] → Critic → Score = 0.55 (FAIL)
    │
    ├── Critic says: "Summary has unsupported claims; Quiz Q3 not answerable from source"
    │
    ▼
Attempt 2: Planner (with feedback) → [4 agents in parallel] → Critic → Score = 0.82 (PASS)
    │
    ▼
  Report compiled ✓
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
| `sentence-transformers` | `from sentence_transformers import SentenceTransformer` | Converts text to 384-dimensional vectors for semantic search |
| `pymupdf` | `import fitz` | Extracts text from PDF pages (note: import name differs from package name!) |
| `faiss-cpu` | `import faiss` | Facebook's library for fast vector similarity search |
| `numpy` | `import numpy as np` | Numerical array operations |
| `python-dotenv` | `from dotenv import load_dotenv` | Loads API keys from `.env` file |

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

This is the **foundation library** that all other files import from. It provides reusable building blocks.

---

#### 4.1.1 LLM Client Setup (Lines 31-44)

```python
PROVIDER = None
_client = None

if os.getenv("GROQ_API_KEY"):
    from groq import Groq
    _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    PROVIDER = "groq"
```

**What it does:** Checks if a Groq API key exists in environment variables. If found, creates a Groq client object that we'll use to make API calls.

**Key concept - Lazy initialization:** The client is created once at import time and reused. This avoids creating a new connection for every API call.

**Key concept - Environment variables:** API keys are stored in `.env` files, not in code. This prevents accidentally committing secrets to Git.

**What students can improve:**
- Add support for multiple providers (OpenAI, Anthropic) with a factory pattern
- Add connection health checks
- Implement connection pooling for high throughput

---

#### 4.1.2 Embedding Model Setup (Lines 50-59)

```python
_embed_model = None

def _get_embed_model():
    """Lazy-load the sentence-transformers embedding model."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model
```

**What it does:** Loads the embedding model the first time it's needed, then reuses it.

**Key concept - Singleton pattern:** The `global _embed_model` ensures only ONE instance of the model exists in memory. Loading a model takes ~2-3 seconds, so we don't want to do it repeatedly.

**Key concept - Lazy loading:** The model isn't loaded when you `import helpers` - only when `embed()` or `build_index()` is first called. This keeps startup fast.

**What is `all-MiniLM-L6-v2`?** A small (22MB), fast embedding model that converts text into 384-dimensional vectors. Similar text produces similar vectors, enabling semantic search. It runs entirely on your CPU - no API calls needed.

**What students can improve:**
- Try different embedding models (e.g., `all-mpnet-base-v2` for higher quality)
- Add GPU support for faster embedding
- Benchmark embedding speed vs. quality tradeoffs

---

#### 4.1.3 Cost Tracking - `CostTracker` Class (Lines 77-176)

```python
class CostTracker:
    def __init__(self, budget=1.00):
        self.budget = budget
        self.calls = []
        self.total_cost = 0.0

    def record(self, llm_result, agent_name="unknown"):
        tokens = llm_result.get("tokens", {})
        model = llm_result.get("model", "llama-3.1-8b-instant")
        pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 3.0})
        cost = (inp * pricing["input"] + out * pricing["output"]) / 1_000_000
        self.total_cost += cost

    def check_budget(self):
        if self.total_cost > self.budget:
            raise RuntimeError(f"Budget exceeded: ${self.total_cost:.4f}")
```

**What it does:** Tracks every LLM API call - which agent made it, how many tokens were used, and the estimated cost. Can enforce a hard budget limit.

**Key concept - Cost awareness:** In production AI systems, LLM calls cost real money. A runaway loop could spend hundreds of dollars. The `check_budget()` method acts as a **circuit breaker** - it stops the pipeline before overspending.

**Key concept - Token counting:** LLMs charge by "tokens" (roughly 3/4 of a word). Every API response includes `input_tokens` (your prompt) and `output_tokens` (the model's response). More tokens = more cost.

**Note:** Groq's free tier currently costs $0.00 per token, so the cost tracking is educational. But the pattern is critical for production systems using paid APIs.

**What students can improve:**
- Add real-time cost alerts (e.g., warn at 80% of budget)
- Track cost per document (for batch processing)
- Add cost prediction before running the pipeline
- Implement a cost dashboard with matplotlib/plotly charts

---

#### 4.1.4 PDF Loading & Chunking (Lines 183-213)

```python
def load_and_chunk(pdf_path, chunk_size=300, overlap=50):
    doc = fitz.open(pdf_path)
    text = ' '.join([page.get_text() for page in doc])
    doc.close()

    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks
```

**What it does:** Opens a PDF, extracts all text, then splits it into overlapping chunks of ~300 words each.

**Key concept - Chunking:** LLMs have token limits. You can't send a 50-page PDF in one prompt. Instead, you split it into chunks and send the most relevant ones.

**Key concept - Overlapping chunks:** The `overlap=50` parameter means consecutive chunks share 50 words. This prevents losing context at chunk boundaries. Without overlap, a sentence split across two chunks would lose its meaning.

**Visual example:**
```
Chunk 1: [word 1 ... word 250 ... word 300]
Chunk 2:                   [word 251 ... word 550]  ← 50 words overlap
Chunk 3:                                [word 501 ... word 800]
```

**Key concept - `step = chunk_size - overlap`:** If chunk_size=300 and overlap=50, step=250. This means each new chunk starts 250 words after the previous one, creating a 50-word overlap.

**What students can improve:**
- Use sentence-aware chunking (don't split mid-sentence)
- Add support for other formats (DOCX, HTML, TXT)
- Implement smart chunking that respects section headers
- Add OCR support for scanned PDFs using `pytesseract`

---

#### 4.1.5 Embedding & Vector Search (Lines 220-256)

```python
def embed(text):
    model = _get_embed_model()
    vec = model.encode(text, convert_to_numpy=True)
    return vec.astype('float32')

def build_index(chunks):
    model = _get_embed_model()
    matrix = model.encode(chunks, convert_to_numpy=True).astype('float32')
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index

def search(index, chunks, query, k=5):
    query_vec = embed(query).reshape(1, -1)
    distances, indices = index.search(query_vec, min(k, index.ntotal))
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": chunks[idx],
            "score": round(float(1 / (1 + dist)), 4),
            "index": int(idx)
        })
    return results
```

**What it does:** Three-step process:
1. `embed()` - Converts text into a 384-dimensional number vector
2. `build_index()` - Embeds ALL chunks and stores them in a FAISS index for fast searching
3. `search()` - Embeds a query, finds the k most similar chunks

**Key concept - Semantic search vs. keyword search:**
- Keyword search: "machine learning" only matches documents containing those exact words
- Semantic search: "machine learning" also matches "ML algorithms", "neural network training", "AI models" because their vectors are similar

**Key concept - FAISS IndexFlatL2:**
- `IndexFlatL2` uses L2 (Euclidean) distance to measure similarity
- Lower distance = more similar
- The score formula `1 / (1 + dist)` converts distance to a 0-1 similarity score

**Key concept - Batch encoding:** `model.encode(chunks)` encodes all chunks at once, which is much faster than encoding one at a time because the model can batch GPU/CPU operations.

**What students can improve:**
- Try `IndexIVFFlat` for faster search on large documents (1000+ chunks)
- Implement hybrid search (semantic + keyword matching with BM25)
- Add metadata filtering (search only within specific sections)
- Use `IndexFlatIP` (inner product) instead of L2 for normalized vectors

---

#### 4.1.6 LLM Calling with Retry & Backoff (Lines 263-347)

```python
def call_llm(prompt, system="You are a helpful assistant.",
             model=None, temperature=0, max_tokens=2000, json_output=False,
             retries=2, backoff_base=2.0):
    last_error = None
    for attempt in range(retries + 1):
        try:
            # ... make API call ...
            response = _client.chat.completions.create(**kwargs)
            return {
                "text": response.choices[0].message.content,
                "tokens": {"input": ..., "output": ...},
                "latency_ms": int(elapsed * 1000),
                "model": m
            }
        except Exception as e:
            last_error = e
            if attempt < retries:
                wait = backoff_base ** attempt  # 1s, 2s, 4s...
                time.sleep(wait)
            else:
                raise last_error
```

**What it does:** Makes an API call to the LLM with automatic retries on failure.

**Key concept - Exponential backoff:** When an API call fails (network error, rate limit, server overload), we don't immediately retry. We wait increasingly longer:
- Attempt 1: fails → wait 1 second
- Attempt 2: fails → wait 2 seconds
- Attempt 3: fails → raise error

This prevents hammering a struggling server and gives it time to recover.

**Key concept - JSON output mode:** When `json_output=True`, two things happen:
1. We append "Return ONLY valid JSON" to the system prompt
2. We set `response_format = {"type": "json_object"}` which forces the model to output valid JSON

**Key concept - `temperature=0`:** Temperature controls randomness. At 0, the model always picks the most likely next token, making outputs deterministic and reproducible. Higher temperatures (0.5-1.0) add creativity but less consistency.

**Two convenience wrappers:**

```python
def call_llm_cheap(...)   # Uses Llama 3.1 8B   → fast, good for planning
def call_llm_strong(...)  # Uses Llama 3.1 8B   → same model, higher max_tokens
```

**Key concept - Model selection strategy:** Use the cheapest model that can do the job. Planners need basic understanding (8B is fine). Summarizers and critics need deeper reasoning (70B).

**What students can improve:**
- Add request/response logging for debugging
- Implement rate limiting (max N requests per minute)
- Add streaming support for real-time output
- Cache responses to avoid duplicate API calls
- Add timeout handling for slow responses

---

#### 4.1.7 Safe JSON Parsing (Lines 354-400)

```python
def parse_json(text):
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in surrounding text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    # Try to find JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    # ...
```

**What it does:** Robustly extracts JSON from LLM output, handling common formatting issues.

**Why is this needed?** LLMs are unreliable at producing clean JSON. They often:
- Wrap JSON in markdown code fences: ` ```json {...} ``` `
- Add explanatory text before/after the JSON
- Return arrays `[...]` instead of objects `{...}`
- Include trailing commas (invalid JSON)

**Key concept - Defense in depth:** The parser tries multiple strategies:
1. Parse directly → works if LLM returned clean JSON
2. Strip markdown fences → works if LLM used code blocks
3. Regex extract `{...}` → works if there's surrounding text
4. Regex extract `[...]` → works if LLM returned an array
5. Raise error → gives up with a clear message

**The `_wrap_if_list` helper:** Some agents return bare arrays instead of objects. This function detects the content type and wraps it in the expected structure:
- `[{"fact": ...}]` → `{"facts": [...], "total_extracted": N}`
- `[{"question": ...}]` → `{"questions": [...]}`

**What students can improve:**
- Add validation with Pydantic models (ensure required fields exist)
- Handle partial JSON (recover from truncated responses)
- Add JSON schema validation against expected structure
- Log parsing failures for debugging

---

#### 4.1.8 Semantic Cache (Lines 407-458)

```python
class SemanticCache:
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.entries = []

    def get(self, query_text):
        query_vec = embed(query_text)
        for cached_text, cached_vec, cached_response in self.entries:
            sim = self._cosine_sim(query_vec, cached_vec)
            if sim >= self.threshold:
                return cached_response
        return None

    def put(self, query_text, response):
        query_vec = embed(query_text)
        self.entries.append((query_text, query_vec, response))
```

**What it does:** Caches LLM responses by semantic similarity. If a new query is very similar (>95%) to a previous query, it returns the cached response instead of making another API call.

**Key concept - Cosine similarity:** Measures the angle between two vectors. A value of 1.0 means identical direction (identical meaning), 0.0 means completely unrelated.

```
"What is machine learning?" ←→ "What is ML?"           → similarity: 0.97 (cache hit!)
"What is machine learning?" ←→ "How does cooking work?" → similarity: 0.12 (cache miss)
```

**Key concept - Threshold tuning:** The 0.95 threshold is conservative - only near-identical queries hit the cache. Lower thresholds (0.85) catch more duplicates but risk returning wrong answers.

**What students can improve:**
- Add TTL (time-to-live) to expire old cache entries
- Implement disk-based caching for persistence across runs
- Add cache hit/miss statistics
- Use FAISS index for the cache itself (faster than linear scan for many entries)
- Implement LRU eviction when cache gets too large

---

#### 4.1.9 Evaluation Harness (Lines 465-608)

```python
class EvalHarness:
    def add_test(self, question, expected_keywords, difficulty="medium"):
        self.test_cases.append({...})

    def run(self, pipeline_fn):
        for test in self.test_cases:
            result = pipeline_fn(test["question"])
            keyword_score = keywords_found / total_keywords
            composite = keyword_score * 0.5 + critic_score * 0.5
            status = "pass" if composite >= 0.6 else "fail"
```

**What it does:** Benchmarks the pipeline against test questions with known answers.

**Key concept - Composite scoring:** Each test is scored on two dimensions:
1. **Keyword score (50%):** Do expected keywords appear in the answer?
2. **Critic score (50%):** Did the Critic agent validate the answer as grounded?

A composite score >= 0.6 = PASS.

**Key concept - Why evaluate?** Without evaluation, you can't tell if changes make the pipeline better or worse. The harness lets you:
- Compare sequential vs. parallel execution quality
- Test if retry logic actually improves scores
- Measure the impact of different chunk sizes or models

**What students can improve:**
- Add LLM-as-judge scoring (use an LLM to grade answer quality)
- Implement A/B testing (compare two pipeline configurations)
- Add regression testing (ensure changes don't break existing quality)
- Track evaluation metrics over time
- Add semantic similarity scoring (not just keyword matching)

---

#### 4.1.10 State Management & Logging (Lines 615-670)

```python
def init_state(query=""):
    return {
        "query": query,
        "chunks": [],
        "log": [],
        "errors": [],
        "start_time": time.time()
    }

def log_agent(state, agent_name, input_summary, output_summary, meta=None):
    entry = {
        "agent": agent_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": input_summary,
        "output": output_summary
    }
    state["log"].append(entry)
```

**What it does:** Provides a shared state dictionary that flows through the pipeline, and structured logging for debugging.

**Key concept - Shared state pattern:** All agents read from and write to the same `state` dictionary. This is simpler than message passing but requires care with parallel execution (each parallel agent gets its own copy).

**Key concept - Structured logging:** Instead of `print("done")`, each agent logs structured data (agent name, timestamp, input/output summary, token count, latency). This makes it easy to debug issues and measure performance.

**What students can improve:**
- Use dataclasses or Pydantic models instead of plain dicts
- Add state snapshots (save state after each agent for debugging)
- Implement event-based logging with Python's `logging` module
- Add tracing IDs for tracking a request through the pipeline

---

### 4.2 `project_intelligence_hub.py` - Complete Solution

This is the **full implementation** that students build towards. It uses everything from `helpers.py`.

---

#### 4.2.1 Agent Prompts (Lines 35-83)

```python
PLANNER_SYSTEM = """You are a Document Planner agent.
Analyze the document and identify structure, themes, and complexity.
Return JSON:
{
    "document_type": "textbook chapter | research paper | ...",
    "main_topic": "one clear sentence",
    "key_themes": ["theme1", "theme2", ...],
    ...
}"""
```

**What it does:** Each agent has a carefully crafted system prompt that defines its role, rules, and expected JSON output format.

**Key concept - Prompt engineering principles used:**

1. **Role assignment:** "You are a [role] agent" tells the LLM to behave as that specialist
2. **Explicit rules:** Numbered, specific rules prevent common mistakes
3. **Output format:** Exact JSON schema ensures parseable responses
4. **Grounding constraint:** "Only extract what is explicitly stated in the chunks" prevents hallucination
5. **Difficulty mixing:** "2 easy, 2 medium, 1 hard" ensures diverse quiz difficulty

**Key concept - Separation of prompts from logic:** Prompts are defined as constants at the top, not embedded in functions. This makes them:
- Easy to find and edit
- Testable independently
- Reusable across different pipeline configurations

**What students can improve:**
- Add few-shot examples to prompts (show the LLM an example output)
- Implement prompt versioning (track which prompt version produced which results)
- A/B test different prompt wordings
- Add chain-of-thought instructions ("Think step by step before answering")

---

#### 4.2.2 Agent Functions (Lines 90-216)

Each agent follows the same pattern:

```python
def agent_name(state, tracker):
    # 1. Extract relevant chunks from state
    chunks_text = "\n---\n".join([c["text"] for c in state["chunks"][:15]])

    # 2. Call LLM with agent-specific prompt
    result = call_llm_strong(
        system=AGENT_SYSTEM_PROMPT,
        prompt=f"Context:\n{chunks_text}",
        json_output=True
    )

    # 3. Track cost
    tracker.record(result, "agent_name")

    # 4. Parse JSON response and store in state
    parsed = parse_json(result["text"])
    state["agent_output"] = parsed

    # 5. Log execution details
    log_agent(state, "agent_name", input_summary, output_summary, meta)

    # 6. Return modified state
    return state
```

**Key concept - Consistent interface:** Every agent takes `(state, tracker)` and returns `state`. This uniformity enables:
- Easy parallel execution (all agents have the same signature)
- Pluggable agents (swap one agent for another without changing the pipeline)
- Testability (each agent can be tested in isolation)

**Planner agent special behavior (adaptive retry):**

```python
def planner(state, tracker):
    retry_context = ""
    if state.get("_critic_feedback"):
        retry_context = f"\n\nPrevious attempt issues: {state['_critic_feedback']}"
    # ... includes retry_context in prompt ...
```

On retry, the Planner reads the Critic's feedback from `state["_critic_feedback"]` and incorporates it into its prompt. This is the **adaptive** part of adaptive retry - the Planner doesn't just re-run blindly, it adjusts based on what went wrong.

---

#### 4.2.3 Parallel Execution with Error Boundaries (Lines 223-267)

```python
def run_parallel_agents(state, tracker):
    agents = {
        "summarizer": summarizer,
        "fact_extractor": fact_extractor,
        "quiz_generator": quiz_generator,
        "gap_analyzer": gap_analyzer
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for name, fn in agents.items():
            agent_state = {**state}  # COPY state for each agent
            futures[executor.submit(fn, agent_state, tracker)] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                result_state = future.result()
                results[name] = result_state
            except Exception as e:
                state["errors"].append({"agent": name, "error": str(e)})
                state[key_map[name]] = {}  # Empty output for failed agent
```

**What it does:** Runs 4 agents simultaneously using threads. Each agent gets its own copy of state to prevent race conditions.

**Key concept - ThreadPoolExecutor:** Python's built-in thread pool. `max_workers=4` means up to 4 threads run concurrently. Since LLM calls are I/O-bound (waiting for network responses), threads work well here despite Python's GIL.

**Key concept - State copying:** `agent_state = {**state}` creates a shallow copy. Each agent gets its own state dict, so they don't overwrite each other's results. After all agents finish, results are merged back.

**Key concept - Error boundaries:** Each agent is wrapped in `try/except`. If the Summarizer crashes, the other three agents still run successfully. The failed agent produces empty output `{}`, and the Critic will score it as 0.

**Key concept - `as_completed(futures)`:** Returns futures as they complete, not in submission order. This means we process results immediately as each agent finishes, rather than waiting for all of them.

**Performance impact:** Sequential execution takes ~12 seconds (3 seconds per agent). Parallel execution takes ~4 seconds (all agents run simultaneously, total time = slowest agent). That's a **3x speedup**.

**What students can improve:**
- Use `asyncio` + `aiohttp` instead of threads for true async I/O
- Add per-agent timeouts (kill agents that take too long)
- Implement agent dependencies (run agent B only after agent A completes)
- Add progress callbacks (update UI as each agent finishes)
- Pool agent results with weighted voting

---

#### 4.2.4 Report Compiler (Lines 274-341)

```python
def report_compiler(state):
    r = []
    r.append("=" * 65)
    r.append("  DOCUMENT INTELLIGENCE REPORT")
    # ... builds formatted report from all agent outputs ...
    state["report"] = "\n".join(r)
    return state
```

**What it does:** Pure Python function (no LLM needed) that assembles all agent outputs into a formatted text report.

**Key concept - No LLM needed:** Not everything needs an LLM. Report formatting is deterministic - the same inputs always produce the same output. Using an LLM here would be slower, more expensive, and less reliable.

**What students can improve:**
- Generate HTML/PDF reports instead of plain text
- Add data visualizations (charts for quality scores, fact distribution)
- Create Markdown output for easy sharing
- Add an interactive web-based report viewer

---

#### 4.2.5 Main Pipeline (Lines 348-424)

```python
def run_pipeline(pdf_path, budget=0.50, max_retries=2, use_cache=True):
    tracker = CostTracker(budget=budget)

    # Step 1: Load + index
    chunks = load_and_chunk(pdf_path)
    index = build_index(chunks)

    # Step 2: Select diverse chunks
    for q in ["main topic overview", "key concept definition",
              "important conclusion", "methodology approach", "example illustration"]:
        all_results.extend(search(index, chunks, q, k=4))
    state["chunks"] = deduplicated_results[:20]

    # Step 3: Planner
    state = planner(state, tracker)

    # Steps 4-5: Analysis + Critic with adaptive retry
    retry_count = 0
    while retry_count <= max_retries:
        tracker.check_budget()
        state = run_parallel_agents(state, tracker)
        state = critic(state, tracker)

        if state["critic_score"] >= 0.7:
            break
        else:
            state = planner(state, tracker)  # Re-plan with feedback
            retry_count += 1

    # Step 6-7: Report + Diagnostics
    state = report_compiler(state)
    tracker.report()
```

**Key concept - Diverse chunk selection:** Instead of taking the first 20 chunks (which would all be from the beginning of the document), we search for 5 different aspects: "overview", "concepts", "conclusion", "methodology", "examples". This ensures chunks from ALL parts of the document are represented.

**Key concept - Deduplication:** The same chunk might match multiple search queries. The `seen` set tracks chunk indices to prevent duplicates:
```python
seen = set()
state["chunks"] = [c for c in all_results
                   if c["index"] not in seen and not seen.add(c["index"])][:20]
```

**Key concept - Retry loop with budget awareness:** Before each retry, `tracker.check_budget()` ensures we have enough budget for another full cycle. This prevents spending money on retries when we can't afford them.

---

#### 4.2.6 Evaluation Mode (Lines 457-491)

```python
def run_evaluation(pdf_path):
    harness = EvalHarness()

    # Auto-generate test questions from the document
    result = call_llm_cheap(
        system="Generate 5 factual questions about this document...",
        prompt=sample_text
    )
    tests = parse_json(result["text"])

    for t in tests["tests"]:
        harness.add_test(question=t["question"], expected_keywords=t["keywords"])

    harness.run(pipeline_for_eval)
    harness.report()
```

**What it does:** Automatically generates test questions from the document, then runs the full pipeline on each question and measures quality.

**Key concept - Auto-generated evaluation:** Instead of manually writing test cases, we use an LLM to generate them. This makes evaluation work on ANY document without manual effort.

**What students can improve:**
- Add human-written test cases for higher quality evaluation
- Implement cross-validation (use different chunks for testing vs. analysis)
- Add regression testing across pipeline versions
- Track evaluation metrics over time with a leaderboard

---

### 4.3 `hub_starter.py` - Student Starter Code

This file provides all the agent functions but leaves the **engineering infrastructure** for students to build.

#### 5 Milestones:

| Milestone | Time | What You Build |
|-----------|------|----------------|
| **1** | 45 min | Wire sequential pipeline (call agents in order) |
| **2** | 45 min | Add parallel execution with ThreadPoolExecutor |
| **3** | 30 min | Add cost tracking and budget enforcement |
| **4** | 30 min | Build evaluation harness with 5 test questions |
| **5** | 30 min | Implement adaptive retry (Critic → Planner feedback) |

**Teaching philosophy:** The agents (AI/ML part) are given. Students build the **software engineering** that makes them work together. This teaches that building AI systems is 80% engineering and 20% ML.

---

## 5. Key Engineering Concepts

### 5.1 Multi-Agent Architecture

**Pattern:** Divide a complex task into specialized agents, each with a single responsibility.

**Benefits:**
- Easier to debug (test each agent independently)
- Agents can run in parallel
- Can swap models per agent (cheap model for planning, expensive for reasoning)
- Failures are isolated

### 5.2 Error Boundaries

**Pattern:** Wrap each agent in try/except so one failure doesn't crash the whole pipeline.

```python
try:
    result = future.result()
except Exception as e:
    state["errors"].append({"agent": name, "error": str(e)})
    state[output_key] = {}  # Empty output
```

**Real-world analogy:** Microservice architecture - if the recommendation service goes down, the shopping cart still works.

### 5.3 Adaptive Retry

**Pattern:** When quality is below threshold, don't retry blindly. Feed failure feedback back into the system.

```
Attempt 1: Planner → Agents → Critic (score: 0.55, feedback: "summary hallucinated")
Attempt 2: Planner (with feedback) → Agents → Critic (score: 0.82, PASS)
```

**vs. Naive retry:** Just re-running with the same inputs usually produces the same output (temperature=0). Adaptive retry changes the inputs based on what went wrong.

### 5.4 Cost Control

**Pattern:** Track every API call's cost and enforce a hard budget limit.

**Why it matters:** A bug that loops infinitely would cost unlimited money with a paid API. The CostTracker acts as a financial circuit breaker.

### 5.5 Semantic Caching

**Pattern:** Cache responses by meaning, not exact string match.

**Why it matters:** "What is ML?" and "What is machine learning?" are different strings but the same question. Semantic caching catches these near-duplicates.

---

## 6. Groq Free Tier Rate Limits

### Understanding the Limits

The Groq free tier imposes these rate limits (the ones that affect us most are marked with **):

| Model | RPM | RPD | **TPM** | **TPD** |
|-------|-----|-----|---------|---------|
| llama-3.1-8b-instant | 30 | 14.4K | **6K** | 500K |
| llama-3.3-70b-versatile | 30 | 1K | **12K** | **100K** |

**Key:**
- **RPM** = Requests Per Minute
- **RPD** = Requests Per Day
- **TPM** = Tokens Per Minute (the main bottleneck!)
- **TPD** = Tokens Per Day (daily budget)

### Why This Matters

Each pipeline run makes approximately:
- 1 call to **8b model** (Planner) = ~2,500 tokens
- 4 calls to **70b model** (analysis agents) = ~3,500 tokens each = ~14,000 tokens
- 1 call to **70b model** (Critic) = ~3,500 tokens
- **Total per run: ~20,000 tokens** (70b model)

With 100K TPD for 70b, students can run the pipeline **~5 times per day**.

### What We Changed (Rate Limiting Adaptations)

| Change | Before | After | Why |
|--------|--------|-------|-----|
| Chunks per analysis agent | 15 | 6 | Cuts input tokens by 60% |
| Chunks for Planner | 8 | 5 | Stays under 8b TPM (6K) |
| Chunks for Critic | 10 | 5 | Reduces 70b token usage |
| max_tokens (70b output) | 8192 | 2000 | Prevents oversized responses |
| Default max_retries | 2 | 1 | Each retry = 5 more API calls |
| Diversity search queries | 5 x k=4 | 3 x k=3 | Fewer chunks needed |
| Inter-call delay | None | 4 seconds | Prevents TPM overflow |
| 429 error backoff | 2-4 seconds | 30+ seconds | Groq needs time to reset |
| Default retries (call_llm) | 2 | 3 | Extra retry for rate limits |

### How the Rate Limiter Works

```python
# In helpers.py - thread-safe rate limiting
_rate_limit_lock = threading.Lock()
_last_llm_call_time = 0
RATE_LIMIT_DELAY = 4  # seconds between calls

# Before every API call:
with _rate_limit_lock:
    elapsed = time.time() - _last_llm_call_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_llm_call_time = time.time()
```

**Why threading.Lock?** When 4 agents run in parallel, they all try to call the API simultaneously. The lock ensures only one thread calls the API at a time, with a 4-second gap between calls. This naturally staggers parallel execution.

### Token Budget Calculator

Use this to plan your lab session:

```
Per pipeline run (no retries):  ~20,000 tokens (70b)
Per pipeline run (1 retry):     ~40,000 tokens (70b)
Daily limit (70b):              100,000 tokens

Runs per day (no retries):      5
Runs per day (with retries):    2-3

Tip: Use --no-cache flag sparingly.
     Keep retries at 1 for lab practice.
```

### Troubleshooting Rate Limit Errors

If you see `[rate limit] Groq free tier limit hit. Waiting 30s...`:
1. **Don't panic** - the code will auto-retry after waiting
2. **Wait a minute** before running again
3. **Check your daily usage** at https://console.groq.com/usage
4. **Reduce retries** with `--retries 0` if you're just testing

---

## 7. How to Run

### Basic Usage

```bash
# Full pipeline on a PDF
python project_intelligence_hub.py sample_docs/cloud_computing.pdf

# With custom budget
python project_intelligence_hub.py sample_docs/neural_networks.pdf --budget 0.30

# With evaluation harness
python project_intelligence_hub.py sample_docs/sustainable_energy.pdf --eval

# All options
python project_intelligence_hub.py your_doc.pdf --budget 0.50 --retries 2 --eval --no-cache
```

### Streamlit UI (Interactive Demo)

```bash
# Install streamlit first
pip install streamlit

# Run the UI
streamlit run streamlit_app.py
```

The Streamlit app provides:
- File upload for any PDF
- Sample document selection
- Real-time progress tracking for each pipeline stage
- Interactive quiz (select answers, check correctness)
- Visual cost & timing breakdown
- Downloadable report

### Starter Code (Learning Mode)

```bash
# Follow milestones 1-5
python hub_starter.py sample_docs/cloud_computing.pdf
```

---

## 8. Streamlit UI Guide

### File: `streamlit_app.py`

The Streamlit app is a **separate, standalone file** that imports from the existing pipeline code. It provides a web-based interface for the Document Intelligence Hub.

### How It Works

```
streamlit_app.py
    ├── imports from helpers.py (load_and_chunk, build_index, etc.)
    ├── imports from project_intelligence_hub.py (agent functions)
    └── provides web UI with:
        ├── File upload / sample selection
        ├── Pipeline configuration sidebar
        ├── Real-time progress bar
        ├── 7-tab results display
        └── Download button for reports
```

### Key UI Components

| Tab | What It Shows |
|-----|---------------|
| **Executive Summary** | Document type, topic, themes, summary text, key points |
| **Key Facts** | Facts grouped by importance (High/Medium/Low) with type badges |
| **Practice Quiz** | Interactive MCQs with radio buttons and answer checking |
| **Gap Analysis** | Coverage score, gaps by severity, recommendations |
| **Quality Validation** | Critic scores per section, issues, improvement hints |
| **Full Report** | Complete text report with download button |
| **Cost & Timing** | Token usage table, timing breakdown, budget status |

### Sidebar Features

- **API Key management** - shows connection status, allows manual key entry
- **Pipeline config** - retries, budget, parallel toggle
- **Rate limit info** - shows current Groq free tier limits
- **Architecture diagram** - quick reference for pipeline flow

### Running the Streamlit App

```bash
# From the intelligence-hub directory:
streamlit run streamlit_app.py

# The app opens at http://localhost:8501
```

### What Students Can Add to the UI

1. **Chat interface** - Add a text input for follow-up questions about the document
2. **Comparison view** - Upload two PDFs and compare analysis side by side
3. **History tab** - Store and display past analysis results
4. **Agent animation** - Show which agent is currently running with animated indicators
5. **Token usage chart** - Use `st.bar_chart()` to visualize token usage per agent

---

## 9. Improvement Ideas for Students

### Beginner Level
1. **Add a new agent** - Create a "Key Terminology" agent that extracts and defines technical terms
2. **Better report formatting** - Generate HTML or Markdown reports instead of plain text
3. **Support more file types** - Add DOCX, TXT, HTML support alongside PDF
4. **Add a progress bar** - Show a progress indicator as each agent completes

### Intermediate Level
5. **Implement streaming** - Show LLM responses in real-time as they generate
6. **Add agent dependencies** - Some agents could use other agents' outputs (e.g., Quiz Generator uses Fact Extractor output)
7. **Hybrid search** - Combine semantic search with keyword search (BM25) for better retrieval
8. **Persistent caching** - Save the semantic cache to disk so it persists across runs
9. **Comparison mode** - Analyze two documents and compare their content

### Advanced Level
10. **Async pipeline** - Replace ThreadPoolExecutor with asyncio for true async I/O
11. **Agent orchestration** - Implement a dynamic agent graph where agents can spawn sub-agents
12. **Multi-document analysis** - Process a folder of related PDFs and cross-reference findings
13. **Custom evaluation metrics** - Implement ROUGE, BLEU, or BERTScore for answer quality
14. **RAG pipeline** - Add a Q&A mode where users can ask follow-up questions about the document
15. **Deployment** - Containerize with Docker and deploy as a web API with FastAPI

### Research Level
16. **Agent collaboration** - Agents discuss and debate findings before producing final output
17. **Self-improving prompts** - Use evaluation results to automatically improve agent prompts
18. **Knowledge graph** - Build a knowledge graph from extracted facts and relationships
19. **Multi-modal analysis** - Extract and analyze images, charts, and tables from PDFs
20. **Benchmark suite** - Create a standardized benchmark to compare different pipeline configurations

---

*This guide was created to help students understand the Multi-Agent Document Intelligence Hub codebase. Each section explains not just WHAT the code does, but WHY it's designed that way and HOW students can improve it.*
