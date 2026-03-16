from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    text: str
    meta: dict
    score: float
    sources: dict  # e.g., {"bm25_rank": 3, "vec_rank": 1}


def reciprocal_rank_fusion(
    bm25_hits: list,
    vec_hits: list,
    k: int = 60,
    top_k: int = 20,
) -> list[Candidate]:
    """
    RRF: score(d) = sum(1 / (k + rank))
    rank is 1-based.
    Produces a merged candidate list.
    """
    merged: Dict[str, Dict[str, Any]] = {}

    def add_hits(hits, key: str):
        for rank0, h in enumerate(hits):
            rank = rank0 + 1
            rrf = 1.0 / (k + rank)
            if h.chunk_id not in merged:
                merged[h.chunk_id] = {
                    "chunk_id": h.chunk_id,
                    "text": h.text,
                    "meta": h.meta,
                    "score": 0.0,
                    "sources": {},
                }
            merged[h.chunk_id]["score"] += rrf
            merged[h.chunk_id]["sources"][f"{key}_rank"] = rank

    add_hits(bm25_hits, "bm25")
    add_hits(vec_hits, "vec")

    items = [
        Candidate(
            chunk_id=v["chunk_id"],
            text=v["text"],
            meta=v["meta"],
            score=float(v["score"]),
            sources=v["sources"],
        )
        for v in merged.values()
    ]
    items.sort(key=lambda c: c.score, reverse=True)
    return items[:top_k]