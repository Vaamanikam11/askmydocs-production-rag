from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional
import os

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer

def _sanitize_meta(meta: dict) -> dict:
    """
    Chroma metadata values must be: str, int, float, bool (or lists of those).
    Convert None -> "" and ensure everything is primitive.
    """
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            # last resort: stringify
            clean[k] = str(v)
    return clean


@dataclass(frozen=True)
class ChromaHit:
    chunk_id: str
    score: float
    text: str
    meta: dict


class ChromaIndex:
    """
    Thin wrapper around Chroma for:
      - indexing chunks (id, text, metadata)
      - vector search (top-k)
    """

    def __init__(self, persist_dir: str, collection_name: str, embedding_model: str):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = SentenceTransformer(embedding_model)

    def reset(self) -> None:
        # delete + recreate collection
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict], batch_size: int = 128) -> None:
        """
        chunks: list of dict rows from chunks.jsonl
        """
        ids: List[str] = []
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for c in chunks:
            ids.append(c["chunk_id"])
            texts.append(c["text"])
            metas.append(
                _sanitize_meta(
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
                )
            )

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_metas = metas[i : i + batch_size]

            embs = self._embedder.encode(batch_texts, normalize_embeddings=True).tolist()
            self._collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metas,
                embeddings=embs,
            )

    def query(self, query: str, top_k: int = 10) -> list[ChromaHit]:
        q_emb = self._embedder.encode([query], normalize_embeddings=True).tolist()
        res = self._collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[ChromaHit] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        # cosine distance in chroma: smaller is closer; convert to similarity-like score
        for cid, text, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - float(dist)
            hits.append(ChromaHit(chunk_id=cid, score=score, text=text, meta=meta))
        return hits