from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import json
from pathlib import Path

from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class BM25Hit:
    chunk_id: str
    score: float
    text: str
    meta: dict


def _tokenize(text: str) -> list[str]:
    # simple tokenizer; we can improve later
    return [t.lower() for t in text.split() if t.strip()]


class BM25Index:
    """
    Build BM25 over chunk texts. Persists minimal artifacts as JSON for simplicity.
    """

    def __init__(self, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._tokenized: list[list[str]] = []

    def build(self, chunks: list[dict]) -> None:
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._texts = [c["text"] for c in chunks]
        self._metas = [
            {
                "doc_id": c["doc_id"],
                "chunk_index": c["chunk_index"],
                "source_path": c["source_path"],
                "source_type": c["source_type"],
                "page": c.get("page"),
                "start_token": c.get("start_token"),
                "end_token": c.get("end_token"),
                "title": c.get("title", ""),
            }
            for c in chunks
        ]
        self._tokenized = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(self._tokenized)

    def save(self) -> None:
        # Persist tokenized corpus + metadata; BM25 is re-created on load.
        payload = {
            "chunk_ids": self._chunk_ids,
            "texts": self._texts,
            "metas": self._metas,
        }
        (self.persist_dir / "bm25.json").write_text(json.dumps(payload), encoding="utf-8")

    def load(self) -> None:
        p = self.persist_dir / "bm25.json"
        payload = json.loads(p.read_text(encoding="utf-8"))
        self._chunk_ids = payload["chunk_ids"]
        self._texts = payload["texts"]
        self._metas = payload["metas"]
        self._tokenized = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(self._tokenized)

    def query(self, query: str, top_k: int = 10) -> list[BM25Hit]:
        if self._bm25 is None:
            raise RuntimeError("BM25Index not built/loaded.")

        q = _tokenize(query)
        scores = self._bm25.get_scores(q)

        # top-k indices
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits: list[BM25Hit] = []
        for i in ranked:
            hits.append(
                BM25Hit(
                    chunk_id=self._chunk_ids[i],
                    score=float(scores[i]),
                    text=self._texts[i],
                    meta=self._metas[i],
                )
            )
        return hits