![CI](https://github.com/Vaamanikam11/askmydocs-production-rag/actions/workflows/ci.yml/badge.svg)

# AskMyDocs — Production RAG (Hybrid Retrieval + Rerank + Citations)

Try it out here: https://askmydocs-rag-project.streamlit.app/

Live demo: <YOUR_STREAMLIT_URL>

Ask questions over uploaded PDFs/Markdown/TXT using a production-style RAG pipeline:
- Hybrid retrieval (BM25 + vector search) + Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Citation enforcement + abstention when evidence is weak
- Evaluation: Golden dataset + RAGAS report
- CI gate: blocks merges if retrieval/citation quality regresses

## Architecture (high level)
1. **Ingest** → parse docs → token-aware chunking w/ overlap  
2. **Index** → BM25 + Chroma (embeddings)
3. **Retrieve** → BM25 + vector → fuse (RRF)
4. **Rerank** → cross-encoder rescoring
5. **Answer** → LLM generates ONLY from sources + citations or abstains

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
streamlit run app/streamlit_app.py

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python3 scripts/ingest.py
streamlit run app/streamlit_app.py

