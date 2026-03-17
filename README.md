# AskMyDocs — Production RAG (Hybrid Retrieval + Rerank + Citations + CI)

Try it out here: https://askmydocs-rag-project.streamlit.app/

![CI](https://github.com/Vaamanikam11/askmydocs-production-rag/actions/workflows/ci.yml/badge.svg)

Domain-specific “Ask My Docs” system with:
- Hybrid retrieval (BM25 + vector)
- Cross-encoder reranking
- Citation enforcement + abstention when evidence is weak
- CI-gated offline evaluation to prevent quality regressions

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python3 scripts/ingest.py
streamlit run app/streamlit_app.py
