# RAG Lab — building and evaluating a RAG system step by step

Hands-on lab where I build a **Retrieval-Augmented Generation (RAG)** system from scratch, then improve it one layer at a time — ingestion, embeddings, vector stores, LangChain, hybrid search — and **measure every change** with an evaluation set.

Built while following a self-directed RAG curriculum (with an AI tutor). Everything runs **locally and for free**: open-source LLM through **Ollama** (`qwen2.5:7b`), multilingual embeddings (`bge-m3`), no paid API.

The test corpus is made of **fictitious documents** written for the exercises (an internship regulation, a solar-company FAQ, a 3-page PDF guide and a **scanned** PDF note), in French, with questions in French, Moroccan Darija (Latin and Arabic script), Arabic and English.

## What's inside

| Step | Files | What it does |
|---|---|---|
| 0. Basics | `tp0_embeddings.py`, `tp0_retriever.py`, `tp0_chat.py` | Embeddings and cosine similarity across languages, a minimal retriever, a first local LLM call |
| 1. RAG by hand | `rag_v1.py` | Chunking, embeddings, top-k search, grounded prompt with citations and refusal |
| 2. Ingestion | `ingestion.py` | PDF parsing (PyMuPDF), **OCR of scanned pages** (RapidOCR), header removal, ligature fixes, section-aware chunking with overlap, tables to Markdown |
| 3. Vector stores | `rag_v3.py`, `faiss_bench.py`, `comparer_embeddings.py` | Persistent **Chroma** index (HNSW, cosine), metadata filters, incremental re-indexing of changed files; **FAISS** benchmark (exact vs HNSW vs IVF: latency / recall trade-off) |
| 4. LangChain | `rag_langchain.py`, `tp4_bases.py`, `tp4_decoupage.py` | Same pipeline rewritten with LCEL, structured output, streaming |
| 5. Advanced retrieval | `recherche.py`, `rag_v4.py`, `transformer_question.py`, `tester_transformations.py` | **Hybrid search** (dense + BM25 with Reciprocal Rank Fusion), multi-query and HyDE query transformations |
| 6. Evaluation | `evaluer_recherche.py`, `evaluer_rag.py`, `evaluer_prompt_v2.py`, `jeu_eval.json`, `resultats_*.json` | 25-question test set (including 3 trap questions whose answer is *not* in the documents), retrieval metrics and an **LLM-as-judge** for correctness and faithfulness |

## Results (saved in `resultats_*.json`, 25 questions)

| Configuration | Context recall (top-4) | Correct answers | Faithfulness | Correct citations | Trap questions refused |
|---|---|---|---|---|---|
| Dense, prompt v1 | 100 % | 76 % | 100 % | 100 % | 3 / 3 |
| **Dense, prompt v2** | **100 %** | **84 %** | 95 % | 90 % | **3 / 3** |
| Hybrid (dense + BM25), prompt v1 | 96 % | 72 % | 97 % | 100 % | 3 / 3 |

Takeaways:
- With good multilingual embeddings (`bge-m3`), retrieval was not the bottleneck: most remaining errors come from **generation** (a 7B model misreading a table, answering too briefly).
- Prompt v2 improved correctness (+8 points) but lowered citation accuracy — a reminder to **re-evaluate everything after each change**.
- On this small corpus, hybrid search did not beat dense search: BM25 helps on exact identifiers but misses Arabic and English questions.

## Run it

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) with the models `qwen2.5:7b` and `bge-m3`.

```bash
uv sync
ollama pull qwen2.5:7b
ollama pull bge-m3
uv run python rag_v3.py        # interactive Q&A over data/ (type @file to filter by document)
uv run python evaluer_rag.py   # full evaluation on jeu_eval.json
```

## Tech stack

Python · Ollama (Qwen2.5 7B) · bge-m3 · Chroma · FAISS · rank-bm25 · sentence-transformers · LangChain (LCEL) · PyMuPDF · RapidOCR

## Next steps

- Reliable RAG: NLI-based answer verification, language control, prompt-injection and poisoning tests
- Agentic RAG with LangGraph (router, corrective RAG, ReAct agent)
- Package structure, tests and CI
