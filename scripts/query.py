from __future__ import annotations

import argparse
import json
from pathlib import Path
import askmydocs.config.env 

from askmydocs.config.settings import get_paths
from askmydocs.index.chroma_store import ChromaIndex
from askmydocs.index.bm25_store import BM25Index
from askmydocs.retrieve.hybrid import reciprocal_rank_fusion
from askmydocs.rerank.cross_encoder import CrossEncoderReranker


def load_chunks_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, required=True, help="Query")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--collection", type=str, default="askmydocs")
    parser.add_argument("--bm25k", type=int, default=10)
    parser.add_argument("--veck", type=int, default=10)
    parser.add_argument("--rrfk", type=int, default=20)
    parser.add_argument("--rerankk", type=int, default=5)
    args = parser.parse_args()

    paths = get_paths()

    bm25 = BM25Index(str(paths.bm25_dir))
    bm25.load()

    chroma = ChromaIndex(
        persist_dir=str(paths.chroma_dir),
        collection_name=args.collection,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    bm25_hits = bm25.query(args.q, top_k=args.bm25k)
    vec_hits = chroma.query(args.q, top_k=args.veck)

    fused = reciprocal_rank_fusion(bm25_hits, vec_hits, top_k=args.rrfk)

    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranked = reranker.rerank(args.q, fused, top_k=args.rerankk)

    print("\n=== Hybrid (RRF) Candidates ===")
    for c in fused[: args.topk]:
        print(f"- rrf={c.score:.4f} id={c.chunk_id} src={c.meta.get('source_path')}")
        print(f"  {c.text[:220]}...\n")

    print("\n=== Reranked (Cross-Encoder) ===")
    for h in reranked[: args.topk]:
        print(f"- score={h.score:.4f} id={h.chunk_id} src={h.meta.get('source_path')}")
        print(f"  {h.text[:220]}...\n")


if __name__ == "__main__":
    main()