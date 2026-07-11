# 🔍 WhatsApp Myth Buster

An agentic RAG system that fact-checks WhatsApp misinformation using multi-step reasoning, hybrid retrieval, semantic caching, and self-reflective evidence grading.

**Live demo:** [Add your Streamlit Cloud URL here]

---

## Overview

WhatsApp forwards are a primary vector for misinformation in India — health myths, financial scams, and political misinformation spread faster than they can be manually fact-checked. This project builds an autonomous agent that takes a raw WhatsApp message, extracts individual factual claims, retrieves relevant evidence, and produces a graded verdict (True / False / Misleading / Unverifiable) with sources and a confidence score — without human intervention.

Rather than a single LLM call, the system is built as a **stateful multi-step agent graph** using LangGraph, where each claim goes through routing, retrieval, reranking, self-reflection, and verdict generation — with automatic retry on insufficient evidence and semantic caching to avoid redundant work on previously-verified claims.

---

## Architecture

```
claim_extractor → memory_check ─┬─(cache hit)──────────────→ memory_updater
                                 └─(cache miss)→ router → hybrid_retriever
                                                    ↑            ↓
                                                    │        reranker
                                                    │            ↓
                                        (insufficient) ← self_reflection
                                                    │            ↓ (sufficient)
                                                    │      verdict_generator
                                                    │            ↓
                                                    └──── memory_updater
                                                              ↓
                                              (more claims) → memory_check
                                              (no more claims) → response_aggregator → END
```

Each claim is processed independently through this graph. On insufficient evidence, the agent retries with a **different retrieval tool** (up to a configurable retry limit) before falling back to a best-effort verdict rather than stalling indefinitely.

---

## Key Design Decisions

**Semantic caching, not string matching.** Before running retrieval on a claim, the agent checks long-term memory (a Qdrant collection of past verdicts) using cosine similarity on embeddings — not exact string match. This means reworded or retyped duplicates of a previously-verified claim ("Eiffel Tower built in 1889" vs. "The Eiffel Tower was built in the year 1889, in Paris") still hit cache, which matters for real WhatsApp forwards that get retyped and reshared constantly.

**Adaptive tool routing.** An LLM-based router (`llama-3.1-8b-instant`, chosen for speed and low cost on high-frequency routing decisions) selects between local fact-check corpus search, live news search, or live scientific search depending on claim type — and is explicitly instructed to avoid repeating the same tool on retry.

**Self-reflection before verdict generation.** Rather than generating a verdict from whatever evidence retrieval returns, a dedicated reflection step scores evidence sufficiency and triggers a retry loop (capped at `MAX_RETRY_COUNT`) if the evidence doesn't adequately support a confident verdict — reducing hallucinated confidence on thin evidence.

**Two-tier model usage.** Claim extraction, routing, and reflection use a fast, cheap model (`llama-3.1-8b-instant`) since these are structural/classification decisions. Final verdict generation uses a stronger model (`llama-3.3-70b-versatile`) since that's the user-facing output where reasoning quality matters most. This cuts token usage substantially versus using the 70B model for every LLM call in the graph.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Agent orchestration | LangGraph (explicit state graph, not a chained pipeline) |
| LLM inference | Groq (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) |
| LLM framework | LangChain |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector database | Qdrant Cloud |
| Keyword search | BM25 (`rank_bm25`) |
| Hybrid retrieval | LangChain `EnsembleRetriever` (BM25 0.4 / dense 0.6) |
| Reranking | `sentence-transformers` CrossEncoder (`BAAI/bge-reranker-base`) |
| Live web search | Tavily API |
| Tracing | LangSmith |
| UI | Streamlit |
| Testing | Pytest (124 tests) |

---

## Notable Engineering Problems Solved

**1. Reranker library incompatibility.** The original implementation used `FlagEmbedding`'s `FlagReranker`, which broke under a newer `transformers` version (`XLMRobertaTokenizer has no attribute 'prepare_for_model'`). Rather than pin an old `transformers` version (which risked breaking the embeddings stack), the reranker was migrated to `sentence-transformers`' `CrossEncoder` — same underlying model (`bge-reranker-base`), stable API, no tokenizer-internals dependency. Manual sigmoid normalization was added since `CrossEncoder.predict()` returns raw logits.

**2. Cosine similarity direction bug in semantic caching.** The cache-lookup function compared similarity scores against a threshold using `if score < 0.15`, under the assumption that Qdrant returned a *distance* (lower = more similar). In practice, LangChain's Qdrant integration with cosine distance returns a *similarity* score (higher = more similar) — so identical claims scored ~0.999, which always failed the `< 0.15` check. This meant caching silently never fired; every claim re-ran the full retrieval pipeline regardless of whether it had already been verified. Found via deliberate cache-hit testing (submitting identical and reworded duplicate claims and inspecting raw scores directly against the vector store), not by code inspection alone — a reminder that mocked unit tests can't catch a bug in the semantics of a real similarity metric.

**3. Groq free-tier rate limiting.** `llama-3.3-70b-versatile`'s free tier (~1,000 requests/day, 100K tokens/day) was quickly exhausted once the full multi-node graph was running repeatedly per test. Resolved by splitting model usage across a fast/cheap model for structural decisions (extraction, routing, reflection) and reserving the 70B model for final verdict generation only — cutting 70B token usage by roughly 3-4x.

---

## Running Locally

```bash
git clone https://github.com/<your-username>/whatsapp-myth-buster.git
cd whatsapp-myth-buster
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # then fill in your API keys
streamlit run ui/app.py --server.fileWatcherType none
```

### Required API keys (free tiers available for all)
- [Groq](https://console.groq.com/keys) — LLM inference
- [Tavily](https://tavily.com) — live web search
- [Qdrant Cloud](https://cloud.qdrant.io) — persistent vector storage

### Running tests
```bash
pytest tests/ -v
```
124 tests covering claim extraction, routing, retrieval, reranking, self-reflection, verdict generation, and edge cases (empty input, LLM failures, exception handling).

---

## Example

**Input:**
> 🚨 Forward to all family groups! Drinking hot water with lemon every morning cures diabetes permanently. WHO has confirmed this.

**Output:**
```json
{
  "claim": "Drinking hot water with lemon every morning cures diabetes permanently",
  "verdict": "False",
  "confidence": 95,
  "explanation": "No provided sources suggest lemon water cures diabetes; lemons may support blood sugar management as part of a balanced diet, not a cure.",
  "sources": ["https://...", "https://..."],
  "cached": false
}
```


## Project Structure

```
whatsapp_myth_buster/
├── nodes/              # LangGraph node implementations
├── tools/               # Hybrid search, web search
├── memory/              # Short-term (LangGraph checkpointer) + long-term (Qdrant) memory
├── data/                # Corpus loading
├── utils/                # Embeddings, Qdrant client
├── ui/                    # Streamlit app                
├── config.py
├── main.py
└── requirements.txt
```