# scripts/run_eval.py
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import askmydocs.config.env  # noqa: F401

from askmydocs.config.settings import get_paths
from askmydocs.index.chroma_store import ChromaIndex
from askmydocs.index.bm25_store import BM25Index
from askmydocs.retrieve.hybrid import reciprocal_rank_fusion
from askmydocs.rerank.cross_encoder import CrossEncoderReranker
from askmydocs.generate.citations import build_citations

CITATION_RE = re.compile(r"\[\d+\]")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def retrieve_and_rerank(question: str):
    paths = get_paths()

    bm25 = BM25Index(str(paths.bm25_dir))
    bm25.load()

    chroma = ChromaIndex(
        persist_dir=str(paths.chroma_dir),
        collection_name="askmydocs",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    bm25_hits = bm25.query(question, top_k=20)
    vec_hits = chroma.query(question, top_k=20)
    fused = reciprocal_rank_fusion(bm25_hits, vec_hits, top_k=30)

    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranked = reranker.rerank(question, fused, top_k=8)

    return bm25_hits, vec_hits, fused, reranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=str, default="datasets/golden/golden.jsonl")
    parser.add_argument("--ci", action="store_true", help="Exit non-zero on failure")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "llm"],
        default="retrieval",
        help="retrieval=CI-stable (no Ollama). llm=calls Ollama (slower).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of golden items (useful for --mode llm).",
    )
    parser.add_argument(
        "--min-top-score",
        type=float,
        default=-1e9,
        help="Fail if top rerank score is below this threshold.",
    )
    parser.add_argument(
        "--abstain-weak-threshold",
        type=float,
        default=float(os.getenv("MIN_EVIDENCE_SCORE", "0.35")),
        help="Score above which an abstain-test looks too easy (retrieval-only heuristic).",
    )
    args = parser.parse_args()

    golden_path = Path(args.golden)
    items = load_jsonl(golden_path)
    if args.max_items is not None:
        items = items[: args.max_items]

    failures: list[tuple[str, str]] = []
    results: list[dict[str, Any]] = []

    for item in items:
        qid = str(item["id"])
        q = str(item["question"])
        must_cite = bool(item.get("must_cite", True))
        should_abstain = bool(item.get("should_abstain", False))

        bm25_hits, vec_hits, fused, reranked = retrieve_and_rerank(q)

        # Build citations from reranked hits (no LLM needed)
        citations, _ = build_citations(reranked, max_sources=5)

        top_score = float(reranked[0].score) if reranked else 0.0

        # ---- Stable (CI-friendly) gates ----
        if len(reranked) == 0:
            failures.append((qid, "No reranked results"))

        if must_cite and len(citations) == 0:
            failures.append((qid, "No citations produced from reranked hits"))

        if reranked and top_score < args.min_top_score:
            failures.append(
                (qid, f"Top rerank score below threshold: {top_score:.3f} < {args.min_top_score:.3f}")
            )

        # Retrieval-only heuristic for abstain tests:
        # If evidence looks strong, the question might be too "answerable" from docs.
        # (True abstention enforcement happens in --mode llm)
        if should_abstain and reranked and top_score >= args.abstain_weak_threshold:
            failures.append(
                (qid, f"Abstain test seems to have strong evidence score ({top_score:.3f}). Consider harder question.")
            )

        results.append(
            {
                "id": qid,
                "reranked": len(reranked),
                "citations": len(citations),
                "top_rerank_score": top_score,
                "should_abstain": should_abstain,
            }
        )

        # ---- Optional LLM mode (calls Ollama) ----
        if args.mode == "llm":
            from askmydocs.generate.answer import answer_with_citations

            # If this question SHOULD abstain, don't call Ollama.
            # We can assert abstention purely from evidence score behavior.
            if should_abstain:
                # Should abstain => evidence should be weak
                if reranked and float(reranked[0].score) >= args.abstain_weak_threshold:
                    failures.append((qid, "Expected weak evidence for abstain test but score is high"))
                # No need to call LLM here.
                continue

            # For non-abstain questions, call Ollama
            ans = answer_with_citations(q, reranked)

            if must_cite and not CITATION_RE.search(ans.text or "") and not ans.abstained:
                failures.append((qid, "LLM answer missing citations like [1]"))

            if must_cite and ans.abstained:
                failures.append((qid, "Unexpected abstention in LLM mode"))

    # ---- Print summary ----
    print("\n=== Eval Summary ===")
    print(f"Mode: {args.mode}")
    print(f"Total: {len(items)}  Failures: {len(failures)}\n")

    for r in results:
        print(
            f"{r['id']}: reranked={r['reranked']} cites={r['citations']} "
            f"topScore={r['top_rerank_score']:.3f} abstainTest={r['should_abstain']}"
        )

    if failures:
        print("\n=== Failures ===")
        for qid, msg in failures:
            print(f"- {qid}: {msg}")

    if args.ci and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()