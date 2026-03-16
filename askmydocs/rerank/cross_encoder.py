from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sentence_transformers import CrossEncoder


@dataclass(frozen=True)
class RerankHit:
    chunk_id: str
    score: float
    text: str
    meta: dict


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list, top_k: int = 8) -> list[RerankHit]:
        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs).tolist()

        hits: list[RerankHit] = []
        for c, s in zip(candidates, scores):
            hits.append(RerankHit(chunk_id=c.chunk_id, score=float(s), text=c.text, meta=c.meta))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]