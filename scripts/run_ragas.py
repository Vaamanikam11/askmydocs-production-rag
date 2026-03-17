from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import askmydocs.config.env  # noqa: F401

from askmydocs.config.settings import get_paths
from askmydocs.index.bm25_store import BM25Index
from askmydocs.index.chroma_store import ChromaIndex
from askmydocs.retrieve.hybrid import reciprocal_rank_fusion
from askmydocs.rerank.cross_encoder import CrossEncoderReranker
from askmydocs.generate.answer import answer_with_citations

# Ragas + LangChain
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def retrieve_contexts(question: str, bm25k: int = 20, veck: int = 20, rrfk: int = 30, rerankk: int = 8, max_sources: int = 5):
    paths = get_paths()

    bm25 = BM25Index(str(paths.bm25_dir))
    bm25.load()

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

    # contexts for Ragas
    contexts = [h.text[:1200] for h in reranked[:max_sources]]
    return reranked, contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=str, default="datasets/golden/golden.jsonl")
    parser.add_argument("--out", type=str, default="artifacts/ragas_report.json")
    parser.add_argument("--max-items", type=int, default=15, help="Limit items for cost/time (recommended)")
    args = parser.parse_args()

    golden = load_jsonl(Path(args.golden))

    # Keep only answerable questions for Ragas (skip abstain tests)
    items = [g for g in golden if not g.get("should_abstain", False)]
    if args.max_items is not None:
        items = items[: args.max_items]

    rows = []
    for it in items:
        q = it["question"]

        reranked, contexts = retrieve_contexts(q)
        ans = answer_with_citations(q, reranked, max_sources=2)  # uses OpenAI provider in your answer.py

        # Ragas dataset schema:
        # question: str
        # answer: str
        # contexts: list[str]
        # ground_truth: str (optional but recommended; we’ll use expected_answer when present)
        rows.append(
            {
                "question": q,
                "answer": ans.text,
                "contexts": contexts,
                "ground_truth": it.get("expected_answer", ""),
                "id": it.get("id", ""),
            }
        )

    ds = Dataset.from_list(rows)

    # Use OpenAI via LangChain for Ragas internal judging
    # (This is separate from your generation model; keep it cheap.)
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.write_text(result.to_pandas().to_json(orient="records"), encoding="utf-8")

    # Print a readable summary
    print("\n=== Ragas Summary ===")
    print(result)
    print(f"\nSaved detailed report to: {out_path}")


if __name__ == "__main__":
    main()