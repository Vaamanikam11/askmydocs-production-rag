from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple
import tiktoken

from askmydocs.config.settings import ChunkingConfig
from askmydocs.ingest.metadata import Chunk, Document, stable_chunk_id


def _get_encoder(name: str):
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        # fallback to a commonly available encoding
        return tiktoken.get_encoding("cl100k_base")


def chunk_document(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    """
    Chunk a single document using token windows with overlap.
    Returns chunks with token offsets so we can later do citation enforcement.
    """
    enc = _get_encoder(cfg.tokenizer_name)
    tokens = enc.encode(doc.text)

    if not tokens:
        return []

    target = cfg.target_tokens
    overlap = cfg.overlap_tokens

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + target, len(tokens))
        window = tokens[start:end]
        text = enc.decode(window).strip()

        # Drop tiny chunks (often headers/footers)
        if len(window) >= cfg.min_tokens and text:
            chunks.append(
                Chunk(
                    chunk_id=stable_chunk_id(doc.doc_id, chunk_index),
                    doc_id=doc.doc_id,
                    chunk_index=chunk_index,
                    text=text,
                    source_path=doc.source_path,
                    source_type=doc.source_type,
                    page=None,  # page-level chunking for PDFs comes later
                    start_token=start,
                    end_token=end,
                    title=doc.title,
                )
            )
            chunk_index += 1

        if end == len(tokens):
            break
        # move forward with overlap
        start = max(0, end - overlap)

    return chunks


def chunk_documents(docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for d in docs:
        all_chunks.extend(chunk_document(d, cfg))
    return all_chunks