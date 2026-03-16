from __future__ import annotations

import time
from pathlib import Path
import json

import streamlit as st

import askmydocs.config.env  # noqa: F401

from askmydocs.config.settings import get_paths
from askmydocs.ingest.loaders import load_documents
from askmydocs.ingest.chunking import chunk_documents
from askmydocs.config.settings import chunking_config_from_env

from askmydocs.index.chroma_store import ChromaIndex
from askmydocs.index.bm25_store import BM25Index

from askmydocs.retrieve.hybrid import reciprocal_rank_fusion
from askmydocs.rerank.cross_encoder import CrossEncoderReranker
from askmydocs.generate.answer import answer_with_citations


ALLOWED_EXTS = {".pdf", ".md", ".markdown", ".txt"}


def save_uploads(uploaded_files, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for uf in uploaded_files:
        p = dest_dir / uf.name
        p.write_bytes(uf.getbuffer())
        saved.append(p)
    return saved


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    files.sort()
    return files


def index_corpus(files: list[Path]) -> dict:
    """
    Full pipeline: load -> chunk -> write chunks.jsonl -> build Chroma + BM25
    """
    paths = get_paths()
    docs = load_documents(files)

    cfg = chunking_config_from_env()
    chunks = chunk_documents(docs, cfg)

    chunks_path = paths.data_processed / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                        "source_path": c.source_path,
                        "source_type": c.source_type,
                        "page": c.page,
                        "start_token": c.start_token,
                        "end_token": c.end_token,
                        "title": c.title,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Build vector index
    chroma = ChromaIndex(
        persist_dir=str(paths.chroma_dir),
        collection_name="askmydocs",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    chroma.reset()
    chroma.add_chunks([json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()])

    # Build BM25 index
    bm25 = BM25Index(str(paths.bm25_dir))
    bm25.build([json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()])
    bm25.save()

    return {
        "files": len(files),
        "docs": len(docs),
        "chunks": len(chunks),
        "chunks_path": str(chunks_path),
    }


def answer_question(question: str, bm25k: int, veck: int, rrfk: int, rerankk: int, minscore: float | None):
    paths = get_paths()

    # Load BM25 index
    bm25 = BM25Index(str(paths.bm25_dir))
    bm25.load()

    # Load vector index
    chroma = ChromaIndex(
        persist_dir=str(paths.chroma_dir),
        collection_name="askmydocs",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    bm25_hits = bm25.query(question, top_k=bm25k)
    vec_hits = chroma.query(question, top_k=veck)
    fused = reciprocal_rank_fusion(bm25_hits, vec_hits, top_k=rrfk)

    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranked = reranker.rerank(question, fused, top_k=rerankk)

    ans = answer_with_citations(question, reranked, min_evidence_score=minscore, max_sources=5)
    return ans, reranked


def main():
    st.set_page_config(page_title="AskMyDocs (Production RAG)", layout="wide")
    st.title("AskMyDocs — Production RAG (Hybrid + Rerank + Citations)")

    paths = get_paths()

    with st.sidebar:
        st.header("Upload & Index")

        uploaded = st.file_uploader(
            "Upload PDFs / Markdown / TXT",
            type=["pdf", "md", "markdown", "txt"],
            accept_multiple_files=True,
        )

        upload_root = paths.data_raw / "uploads"
        st.caption(f"Uploads saved under: {upload_root}")

        if st.button("Save uploads + Re-index", type="primary", disabled=(not uploaded)):
            ts = time.strftime("%Y%m%d-%H%M%S")
            dest = upload_root / ts
            saved_paths = save_uploads(uploaded, dest)

            st.write(f"Saved {len(saved_paths)} file(s). Indexing…")
            with st.spinner("Loading → chunking → indexing (Chroma + BM25)…"):
                # Index ONLY the uploaded batch folder to keep demo clean
                files = iter_files(dest)
                stats = index_corpus(files)

            st.success(f"Indexed: files={stats['files']}, docs={stats['docs']}, chunks={stats['chunks']}")
            st.session_state["indexed"] = True

        st.divider()
        st.header("Retrieval Settings")
        bm25k = st.slider("BM25 top-k", 5, 50, 20, 5)
        veck = st.slider("Vector top-k", 5, 50, 20, 5)
        rrfk = st.slider("Hybrid candidates (RRF top-k)", 5, 80, 30, 5)
        rerankk = st.slider("Rerank top-k", 3, 15, 8, 1)

        st.divider()
        st.header("Citation Enforcement")
        minscore = st.number_input(
            "Min evidence score (cross-encoder). If top score < this, abstain.",
            value=float(0.0),
            step=0.1,
        )
        show_debug = st.checkbox("Show reranked chunks (debug)", value=False)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("Ask a question")
        q = st.text_input("Question", placeholder="e.g., What is hybrid retrieval and why use it?")

        ask_clicked = st.button("Ask", disabled=(not q.strip()))
        if ask_clicked:
            try:
                ans, reranked = answer_question(
                    question=q.strip(),
                    bm25k=bm25k,
                    veck=veck,
                    rrfk=rrfk,
                    rerankk=rerankk,
                    minscore=minscore,
                )

                st.markdown("### Answer")
                st.write(ans.text)

                if ans.citations_block:
                    st.markdown("### Sources")
                    st.code(ans.citations_block)

                st.caption(f"abstained={ans.abstained} | evidence_score={ans.evidence_score:.4f}")

                if show_debug:
                    st.markdown("### Debug: Top reranked chunks")
                    for i, h in enumerate(reranked, start=1):
                        meta = h.meta or {}
                        st.markdown(f"**#{i} score={h.score:.4f}** — `{meta.get('source_path','')}`")
                        st.write(h.text[:800] + ("..." if len(h.text) > 800 else ""))
                        st.divider()

            except FileNotFoundError:
                st.error("Indexes not found. Upload docs and click 'Save uploads + Re-index' first.")
            except Exception as e:
                st.exception(e)

    with col2:
        st.subheader("Index status")
        chunks_path = paths.data_processed / "chunks.jsonl"
        if chunks_path.exists():
            st.success("chunks.jsonl found ✅")
            st.caption(str(chunks_path))
        else:
            st.warning("No chunks.jsonl yet. Upload docs and re-index.")

        st.subheader("Tips")
        st.write(
            "- Start by uploading 1–3 documents.\n"
            "- Ask a question using terms from the docs.\n"
            "- Turn on *Show reranked chunks* to validate grounding.\n"
            "- If it abstains too often, lower *Min evidence score*."
        )


if __name__ == "__main__":
    main()