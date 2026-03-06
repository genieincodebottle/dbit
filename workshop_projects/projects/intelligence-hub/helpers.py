"""
Shared Helper Utilities for Multi-Agent Workshop
================================================
Building blocks used by ALL labs and projects:
  - PDF loading and chunking
  - Embedding (text -> vector) using open-source sentence-transformers (free, local)
  - FAISS index (build + search)
  - LLM calling (Groq) with retry + backoff
  - Cost tracking (real-time token/dollar estimates)
  - Evaluation harness (benchmark your pipeline)
  - State management + structured logging
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import fitz  # PyMuPDF - install with: pip install pymupdf
import faiss
import numpy as np
import json
import time
import re
import os
import hashlib
import threading
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# RATE LIMITING (Groq Free Tier)
# ============================================================
# Free tier limits (from console.groq.com):
#   llama-3.1-8b-instant:    RPM=30, RPD=14.4K, TPM=6K,  TPD=500K
#   llama-3.3-70b-versatile: RPM=30, RPD=1K,    TPM=12K, TPD=100K
#
# Using 8b model only to avoid 70b limits (1K RPD, 100K TPD).
# 8b has generous limits: 14.4K RPD, 500K TPD - plenty for labs.
# Strategy: enforce delay between calls + keep max_tokens low + handle 429 gracefully.

_rate_limit_lock = threading.Lock()
_last_llm_call_time = 0
RATE_LIMIT_DELAY = 5  # seconds between API calls (keeps us safely under TPM limits)

# ============================================================
# LLM CLIENT SETUP (Groq - free tier)
# ============================================================

PROVIDER = None
_client = None

if os.getenv("GROQ_API_KEY"):
    from groq import Groq
    _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    PROVIDER = "groq"
    print("[helpers] Using Groq API (free tier)")
else:
    print("[helpers] WARNING: No API key found. Set GROQ_API_KEY in .env")

# ============================================================
# EMBEDDING MODEL SETUP (local, open-source, free)
# ============================================================

_embed_model = None

def _get_embed_model():
    """Lazy-load the sentence-transformers embedding model."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[helpers] Loaded local embedding model: all-MiniLM-L6-v2 (free, 384 dims)")
    return _embed_model


# ============================================================
# COST TRACKING
# ============================================================

# Pricing per 1M tokens (USD) - Groq free tier = $0.00
MODEL_PRICING = {
    "llama-3.1-8b-instant":       {"input": 0.00,  "output": 0.00},
    "llama-3.3-70b-versatile":    {"input": 0.00,  "output": 0.00},
    "llama3-70b-8192":            {"input": 0.00,  "output": 0.00},
    "gemma2-9b-it":               {"input": 0.00,  "output": 0.00},
    "mixtral-8x7b-32768":         {"input": 0.00,  "output": 0.00},
    "local-embedding":            {"input": 0.00,  "output": 0.00},
}


class CostTracker:
    """
    Tracks token usage and estimated cost across all LLM calls.

    Usage:
        tracker = CostTracker(budget=0.50)  # $0.50 budget

        result = call_llm(...)
        tracker.record(result)              # auto-extracts tokens + model

        tracker.report()                    # prints breakdown
        tracker.remaining()                 # dollars left
        tracker.check_budget()              # raises if over budget
    """

    def __init__(self, budget=1.00):
        self.budget = budget
        self.calls = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def record(self, llm_result, agent_name="unknown"):
        """Record a call_llm() result. Extracts tokens and estimates cost."""
        tokens = llm_result.get("tokens", {})
        model = llm_result.get("model", "llama-3.1-8b-instant")
        inp = tokens.get("input", 0)
        out = tokens.get("output", 0)

        pricing = MODEL_PRICING.get(model, {"input": 1.0, "output": 3.0})
        cost = (inp * pricing["input"] + out * pricing["output"]) / 1_000_000

        self.calls.append({
            "agent": agent_name,
            "model": model,
            "input_tokens": inp,
            "output_tokens": out,
            "cost": cost,
            "latency_ms": llm_result.get("latency_ms", 0)
        })
        self.total_input_tokens += inp
        self.total_output_tokens += out
        self.total_cost += cost
        return cost

    def remaining(self):
        """Dollars remaining in budget."""
        return max(0, self.budget - self.total_cost)

    def check_budget(self):
        """Raise RuntimeError if budget exceeded."""
        if self.total_cost > self.budget:
            raise RuntimeError(
                f"Budget exceeded: ${self.total_cost:.4f} spent of ${self.budget:.2f} budget. "
                f"Total tokens: {self.total_input_tokens} in / {self.total_output_tokens} out"
            )

    def report(self):
        """Print a detailed cost breakdown by agent and model."""
        print("\n" + "=" * 65)
        print("  COST REPORT")
        print("=" * 65)

        # Per-agent breakdown
        agent_costs = {}
        for call in self.calls:
            name = call["agent"]
            if name not in agent_costs:
                agent_costs[name] = {"calls": 0, "tokens": 0, "cost": 0, "latency": 0}
            agent_costs[name]["calls"] += 1
            agent_costs[name]["tokens"] += call["input_tokens"] + call["output_tokens"]
            agent_costs[name]["cost"] += call["cost"]
            agent_costs[name]["latency"] += call["latency_ms"]

        print(f"\n  {'Agent':<22} {'Calls':>5} {'Tokens':>8} {'Cost':>9} {'Latency':>9}")
        print(f"  {'-' * 57}")
        for name, data in sorted(agent_costs.items(), key=lambda x: -x[1]["cost"]):
            print(f"  {name:<22} {data['calls']:>5} {data['tokens']:>8} ${data['cost']:>7.4f} {data['latency']:>7}ms")

        print(f"  {'-' * 57}")
        print(f"  {'TOTAL':<22} {len(self.calls):>5} "
              f"{self.total_input_tokens + self.total_output_tokens:>8} "
              f"${self.total_cost:>7.4f} "
              f"{sum(c['latency_ms'] for c in self.calls):>7}ms")

        print(f"\n  Budget: ${self.budget:.2f} | Spent: ${self.total_cost:.4f} | "
              f"Remaining: ${self.remaining():.4f}")
        print("=" * 65)

    def to_dict(self):
        """Export as dict for serialization or state logging."""
        return {
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "budget_usd": self.budget,
            "remaining_usd": round(self.remaining(), 6),
            "calls": self.calls
        }


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def load_and_chunk(pdf_path, chunk_size=300, overlap=50):
    """
    Load a PDF and split into overlapping text chunks.

    Args:
        pdf_path:   Path to a PDF file
        chunk_size: Number of words per chunk (default 300)
        overlap:    Number of overlapping words between chunks (default 50)

    Returns:
        List of text strings, each ~chunk_size words
    """
    doc = fitz.open(pdf_path)
    pages = len(doc)
    text = ' '.join([page.get_text() for page in doc])
    doc.close()

    words = text.split()
    if not words:
        raise ValueError(f"No text found in {pdf_path}. Is it a scanned PDF? (scans need OCR)")

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    print(f"  Loaded {pages} pages, {len(words)} words -> {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={overlap})")
    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def embed(text):
    """
    Convert text into a vector embedding using a local open-source model.
    Uses sentence-transformers (all-MiniLM-L6-v2) - completely free, no API calls.
    Similar meanings produce similar vectors, enabling semantic search.
    """
    model = _get_embed_model()
    vec = model.encode(text, convert_to_numpy=True)
    return vec.astype('float32')


def build_index(chunks):
    """Build a FAISS index from text chunks. Returns the searchable index."""
    print(f"  Embedding {len(chunks)} chunks (local model - no API calls)...")
    model = _get_embed_model()
    # Batch encode all chunks at once for efficiency (local model, no rate limits)
    matrix = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False).astype('float32')
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    print(f"  Index ready: {index.ntotal} vectors, {matrix.shape[1]} dimensions")
    return index


def search(index, chunks, query, k=5):
    """Search FAISS index for the k most similar chunks to query."""
    query_vec = embed(query).reshape(1, -1)
    distances, indices = index.search(query_vec, min(k, index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(chunks):
            results.append({
                "text": chunks[idx],
                "score": round(float(1 / (1 + dist)), 4),
                "index": int(idx)
            })
    return results


# ============================================================
# LLM CALLING - WITH RETRY + BACKOFF
# ============================================================

def call_llm(prompt, system="You are a helpful assistant.",
             model=None, temperature=0, max_tokens=1024, json_output=False,
             retries=1, backoff_base=2.0):
    """
    Call the LLM with automatic retry on transient failures.
    Includes rate-limit-aware retry (429 errors get 30s+ backoff).

    Retry strategy (exponential backoff):
      Attempt 1: immediate
      Attempt 2: wait 2s
      Attempt 3: wait 4s
      Attempt 4: wait 8s (or 30s for rate limits)

    Args:
        prompt, system, model, temperature, max_tokens, json_output: standard LLM params
        retries:      Number of retry attempts on failure (default 2)
        backoff_base: Base seconds for exponential backoff (default 2.0)

    Returns:
        Dict: {"text", "tokens": {"input", "output"}, "latency_ms", "model"}

    Raises:
        Last exception if all retries exhausted.
    """
    last_error = None

    for attempt in range(retries + 1):
        try:
            # Rate limiting: enforce minimum delay between API calls
            # This prevents hitting Groq free tier TPM limits
            global _last_llm_call_time
            with _rate_limit_lock:
                now = time.time()
                elapsed = now - _last_llm_call_time
                if elapsed < RATE_LIMIT_DELAY:
                    time.sleep(RATE_LIMIT_DELAY - elapsed)
                _last_llm_call_time = time.time()

            start = time.time()

            if PROVIDER == "groq":
                m = model or "llama-3.1-8b-instant"
                sys_prompt = system
                if json_output:
                    sys_prompt += "\n\nIMPORTANT: Return ONLY valid JSON. No explanation, no markdown code fences."

                kwargs = {
                    "model": m,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt}
                    ]
                }
                if json_output:
                    kwargs["response_format"] = {"type": "json_object"}

                response = _client.chat.completions.create(**kwargs)
                elapsed = time.time() - start

                return {
                    "text": response.choices[0].message.content,
                    "tokens": {
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens
                    },
                    "latency_ms": int(elapsed * 1000),
                    "model": m
                }
            else:
                raise RuntimeError("No API client configured. Set GROQ_API_KEY in .env")

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Handle rate limit errors (429) with longer backoff
            if "rate_limit" in error_msg or "429" in error_msg or "too many" in error_msg:
                wait = max(30, backoff_base ** (attempt + 3))
                print(f"    [rate limit] Groq free tier limit hit. Waiting {wait:.0f}s before retry...")
                time.sleep(wait)
            elif attempt < retries:
                wait = backoff_base ** attempt
                print(f"    [retry] Attempt {attempt + 1} failed: {e}. Waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise last_error


def call_llm_cheap(prompt, system="You are a helpful assistant.",
                   temperature=0, max_tokens=500, json_output=False):
    """Fast model: Llama 3.1 8B on Groq (free). Use for planners and simple routing.
    max_tokens=500 (planner JSON is small)."""
    model = "llama-3.1-8b-instant"
    return call_llm(prompt, system, model=model,
                    temperature=temperature, max_tokens=max_tokens, json_output=json_output)


def call_llm_strong(prompt, system="You are a helpful assistant.",
                    temperature=0, max_tokens=1024, json_output=False):
    """Strong model: Also Llama 3.1 8B on Groq (free). Use for reasoning, analysis, judgment.
    Using 8b for all calls to avoid 70b limits (1K RPD, 100K TPD).
    8b has 14.4K RPD and 500K TPD - plenty for lab use across multiple projects.
    Students can swap to llama-3.3-70b-versatile if they have a paid plan."""
    model = "llama-3.1-8b-instant"
    return call_llm(prompt, system, model=model,
                    temperature=temperature, max_tokens=max_tokens, json_output=json_output)


# ============================================================
# JSON PARSING (SAFE)
# ============================================================

def parse_json(text):
    """
    Safely parse JSON from LLM output.
    Handles markdown code blocks, surrounding text, and common formatting issues.
    Wraps bare arrays in a dict with an auto-detected key (LLMs sometimes return arrays).
    """
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())

    def _wrap_if_list(parsed):
        """Wrap bare arrays in expected dict structure based on content."""
        if not isinstance(parsed, list):
            return parsed
        if not parsed or not isinstance(parsed[0], dict):
            return {"items": parsed}
        first = parsed[0]
        if "fact" in first:
            return {"facts": parsed, "total_extracted": len(parsed)}
        if "question" in first:
            return {"questions": parsed}
        if "topic" in first and "severity" in first:
            return {"gaps": parsed, "coverage_score": 0.75, "recommendation": "See gaps above."}
        if "score" in first or "overall_score" in first:
            return parsed[0] if len(parsed) == 1 else {"items": parsed}
        return {"items": parsed}

    try:
        return _wrap_if_list(json.loads(text))
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return _wrap_if_list(json.loads(match.group()))
        except json.JSONDecodeError:
            pass

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return _wrap_if_list(json.loads(match.group()))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output:\n{text[:300]}...")


# ============================================================
# SEMANTIC CACHE
# ============================================================

class SemanticCache:
    """
    Cache LLM responses by query similarity.

    If a new query is very similar to a cached query (cosine similarity > threshold),
    return the cached response instead of calling the LLM again.

    This saves real money: a repeated or near-identical question costs $0 instead of $0.01+.

    Usage:
        cache = SemanticCache(threshold=0.95)

        # Check before calling LLM
        cached = cache.get(query)
        if cached:
            return cached

        # If cache miss, call LLM and store
        result = call_llm(query)
        cache.put(query, result)
    """

    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.entries = []  # list of (query_text, query_vec, response)

    def _cosine_sim(self, a, b):
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    def get(self, query_text, query_vec=None):
        """Check if a similar query exists in cache. Returns cached response or None."""
        if not self.entries:
            return None
        if query_vec is None:
            query_vec = embed(query_text)

        for cached_text, cached_vec, cached_response in self.entries:
            sim = self._cosine_sim(query_vec, cached_vec)
            if sim >= self.threshold:
                return cached_response
        return None

    def put(self, query_text, response, query_vec=None):
        """Store a query-response pair in cache."""
        if query_vec is None:
            query_vec = embed(query_text)
        self.entries.append((query_text, query_vec, response))

    def stats(self):
        return {"entries": len(self.entries)}


# ============================================================
# EVALUATION HARNESS
# ============================================================

class EvalHarness:
    """
    Benchmark your pipeline against a set of test cases with known answers.

    Usage:
        harness = EvalHarness()
        harness.add_test(
            question="What is the return policy?",
            expected_keywords=["30 days", "receipt", "refund"],
            expected_answer="Returns accepted within 30 days with original receipt."
        )
        harness.add_test(
            question="Who is the CEO?",
            expected_keywords=["Sarah Chen"],
        )

        # Run your pipeline on all test cases
        results = harness.run(my_pipeline_fn)
        harness.report()
    """

    def __init__(self):
        self.test_cases = []
        self.results = []

    def add_test(self, question, expected_keywords=None, expected_answer=None,
                 difficulty="medium"):
        """
        Add a test case.

        Args:
            question:          The query to test
            expected_keywords: List of strings that MUST appear in the answer
            expected_answer:   Optional full expected answer (for LLM-as-judge scoring)
            difficulty:        easy/medium/hard (for reporting)
        """
        self.test_cases.append({
            "question": question,
            "expected_keywords": expected_keywords or [],
            "expected_answer": expected_answer,
            "difficulty": difficulty
        })

    def run(self, pipeline_fn):
        """
        Run pipeline_fn(question) for each test case and evaluate results.

        pipeline_fn should accept a question string and return a dict with at least:
          {"answer": "...", "critic_score": 0.85, ...}
        """
        self.results = []
        print(f"\n  Running {len(self.test_cases)} evaluation tests...")
        print(f"  {'#':<3} {'Difficulty':<10} {'Keywords':>8} {'Critic':>7} {'Latency':>9} Question")
        print(f"  {'-' * 75}")

        for i, test in enumerate(self.test_cases, 1):
            start = time.time()
            try:
                result = pipeline_fn(test["question"])
                elapsed = time.time() - start

                answer = result.get("answer", result.get("report", ""))
                critic_score = result.get("critic_score", 0)

                # Check keyword presence
                answer_lower = answer.lower() if isinstance(answer, str) else str(answer).lower()
                keywords_found = sum(
                    1 for kw in test["expected_keywords"]
                    if kw.lower() in answer_lower
                )
                keyword_score = keywords_found / len(test["expected_keywords"]) if test["expected_keywords"] else 1.0

                # Composite score
                composite = (keyword_score * 0.5 + critic_score * 0.5)

                test_result = {
                    "question": test["question"],
                    "difficulty": test["difficulty"],
                    "keyword_score": round(keyword_score, 2),
                    "keywords_found": keywords_found,
                    "keywords_total": len(test["expected_keywords"]),
                    "critic_score": round(critic_score, 2),
                    "composite_score": round(composite, 2),
                    "latency_s": round(elapsed, 1),
                    "status": "pass" if composite >= 0.6 else "fail",
                    "error": None
                }
            except Exception as e:
                elapsed = time.time() - start
                test_result = {
                    "question": test["question"],
                    "difficulty": test["difficulty"],
                    "keyword_score": 0, "keywords_found": 0,
                    "keywords_total": len(test["expected_keywords"]),
                    "critic_score": 0, "composite_score": 0,
                    "latency_s": round(elapsed, 1),
                    "status": "error",
                    "error": str(e)
                }

            self.results.append(test_result)
            status_icon = "PASS" if test_result["status"] == "pass" else "FAIL" if test_result["status"] == "fail" else "ERR "
            print(f"  {i:<3} {test['difficulty']:<10} "
                  f"{test_result['keywords_found']}/{test_result['keywords_total']:>3}    "
                  f"{test_result['critic_score']:>5.2f}  "
                  f"{test_result['latency_s']:>7.1f}s  "
                  f"[{status_icon}] {test['question'][:40]}")

        return self.results

    def report(self):
        """Print evaluation summary."""
        if not self.results:
            print("  No results yet. Call harness.run() first.")
            return

        passed = sum(1 for r in self.results if r["status"] == "pass")
        failed = sum(1 for r in self.results if r["status"] == "fail")
        errors = sum(1 for r in self.results if r["status"] == "error")
        total = len(self.results)

        avg_composite = sum(r["composite_score"] for r in self.results) / total
        avg_latency = sum(r["latency_s"] for r in self.results) / total

        print(f"\n  {'=' * 50}")
        print(f"  EVALUATION RESULTS")
        print(f"  {'=' * 50}")
        print(f"  Tests:     {total} total | {passed} passed | {failed} failed | {errors} errors")
        print(f"  Pass rate: {passed/total*100:.0f}%")
        print(f"  Avg score: {avg_composite:.2f}")
        print(f"  Avg time:  {avg_latency:.1f}s per query")

        # By difficulty
        for diff in ["easy", "medium", "hard"]:
            subset = [r for r in self.results if r["difficulty"] == diff]
            if subset:
                p = sum(1 for r in subset if r["status"] == "pass")
                print(f"    {diff:>6}: {p}/{len(subset)} passed")

        print(f"  {'=' * 50}")

    def to_dict(self):
        """Export results for JSON serialization."""
        return {"test_cases": len(self.test_cases), "results": self.results}


# ============================================================
# STATE MANAGEMENT
# ============================================================

def init_state(query=""):
    """Initialize the shared state dictionary."""
    return {
        "query": query,
        "chunks": [],
        "log": [],
        "errors": [],
        "start_time": time.time()
    }


def log_agent(state, agent_name, input_summary, output_summary, meta=None):
    """Add a structured log entry for an agent execution."""
    entry = {
        "agent": agent_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": input_summary,
        "output": output_summary
    }
    if meta:
        entry.update(meta)
    state["log"].append(entry)
    return state


def print_log(state):
    """Pretty-print the agent execution log."""
    print("\n" + "=" * 65)
    print("  AGENT EXECUTION LOG")
    print("=" * 65)

    total_in, total_out = 0, 0
    for entry in state["log"]:
        print(f"\n  [{entry['agent'].upper()}] @ {entry.get('timestamp', '?')}")
        if 'tokens' in entry:
            t = entry['tokens']
            inp, out = t.get('input', 0), t.get('output', 0)
            total_in += inp
            total_out += out
            print(f"    Tokens: {inp} in / {out} out")
        if 'latency_ms' in entry:
            print(f"    Latency: {entry['latency_ms']}ms")
        output = entry.get('output', '')
        if isinstance(output, str):
            print(f"    Output: {output[:120]}{'...' if len(output) > 120 else ''}")

    elapsed = time.time() - state.get("start_time", time.time())
    print(f"\n  {'-' * 45}")
    print(f"  Total tokens: {total_in} in / {total_out} out")
    print(f"  Total time:   {elapsed:.1f}s")
    print(f"  Agents called: {len(state['log'])}")
    if state.get("errors"):
        print(f"  Errors: {len(state['errors'])}")
        for err in state["errors"]:
            print(f"    [{err.get('agent', '?')}] {err.get('error', '?')}")
    print("=" * 65)


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    print("\n--- Testing helpers.py ---\n")

    print("1. Testing LLM call with retry...")
    result = call_llm_cheap("Say 'Hello Workshop!' and nothing else.")
    print(f"   Response: {result['text']}")

    print("\n2. Testing CostTracker...")
    tracker = CostTracker(budget=0.50)
    tracker.record(result, agent_name="test")
    tracker.report()

    print("\n3. Testing JSON output + parse...")
    result = call_llm_cheap("Return JSON: {\"status\": \"ok\"}", json_output=True)
    parsed = parse_json(result['text'])
    print(f"   Parsed: {parsed}")

    print("\n4. Testing embedding...")
    vec = embed("machine learning is great")
    print(f"   Vector shape: {vec.shape}")

    print("\n5. Testing SemanticCache...")
    cache = SemanticCache(threshold=0.95)
    cache.put("What is ML?", {"answer": "Machine learning is..."})
    hit = cache.get("What is machine learning?")
    print(f"   Cache hit: {hit is not None}")

    print("\n--- All tests passed. Ready for labs. ---")
