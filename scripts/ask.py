from __future__ import annotations

import argparse
import askmydocs.config.env 

from askmydocs.config.settings import get_paths
from askmydocs.index.chroma_store import ChromaIndex
from askmydocs.index.bm25_store import BM25Index
from askmydocs.retrieve.hybrid import reciprocal_rank_fusion
from askmydocs.rerank.cross_encoder import CrossEncoderReranker
from askmydocs.generate.answer import answer_with_citations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, required=True, help="Question")
    parser.add_argument("--bm25k", type=int, default=20)
    parser.add_argument("--veck", type=int, default=20)
    parser.add_argument("--rrfk", type=int, default=30)
    parser.add_argument("--rerankk", type=int, default=8)
    parser.add_argument("--minscore", type=float, default=None)
    args = parser.parse_args()

    paths = get_paths()

    bm25 = BM25Index(str(paths.bm25_dir))
    bm25.load()

    chroma = ChromaIndex(
        persist_dir=str(paths.chroma_dir),
        collection_name="askmydocs",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    bm25_hits = bm25.query(args.q, top_k=args.bm25k)
    vec_hits = chroma.query(args.q, top_k=args.veck)
    fused = reciprocal_rank_fusion(bm25_hits, vec_hits, top_k=args.rrfk)

    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranked = reranker.rerank(args.q, fused, top_k=args.rerankk)

    ans = answer_with_citations(
        question=args.q,
        reranked_hits=reranked,
        min_evidence_score=args.minscore,
        max_sources=5,
    )

    print("\n=== Answer ===")
    print(ans.text)
    if ans.citations_block:
        print(ans.citations_block)
    print(f"\n(abstained={ans.abstained}, evidence_score={ans.evidence_score:.4f})")


if __name__ == "__main__":
    main()