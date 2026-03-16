from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib


def stable_doc_id(source_path: Path) -> str:
    """
    Stable ID based on file path (relative-ish) + size + mtime.
    Good enough for local pipelines; can be swapped to content-hash later.
    """
    try:
        stat = source_path.stat()
        payload = f"{source_path.as_posix()}::{stat.st_size}::{int(stat.st_mtime)}"
    except FileNotFoundError:
        payload = source_path.as_posix()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stable_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}::c{chunk_index:05d}"


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_path: str
    title: str
    text: str
    # Optional structured info:
    source_type: str  # "pdf" | "markdown" | "text" | ...


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str

    # Citation metadata:
    source_path: str
    source_type: str
    page: int | None
    start_token: int
    end_token: int

    # Helpful for UI/debug:
    title: str