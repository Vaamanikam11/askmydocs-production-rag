from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import re

from pypdf import PdfReader

from askmydocs.ingest.metadata import Document, stable_doc_id


_WS_RE = re.compile(r"\s+")


def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s


def load_pdf(path: Path) -> Document:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = _clean_text(txt)
        if txt:
            parts.append(txt)
    full = "\n\n".join(parts).strip()

    title = path.stem
    return Document(
        doc_id=stable_doc_id(path),
        source_path=str(path),
        title=title,
        text=full,
        source_type="pdf",
    )


def load_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # Keep markdown as text; later we can add MD-to-text conversion if needed.
    text = _clean_text(raw)
    return Document(
        doc_id=stable_doc_id(path),
        source_path=str(path),
        title=path.stem,
        text=text,
        source_type="markdown",
    )


def load_text(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = _clean_text(raw)
    return Document(
        doc_id=stable_doc_id(path),
        source_path=str(path),
        title=path.stem,
        text=text,
        source_type="text",
    )


def load_documents(paths: Iterable[Path]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        ext = p.suffix.lower()
        if ext == ".pdf":
            docs.append(load_pdf(p))
        elif ext in (".md", ".markdown"):
            docs.append(load_markdown(p))
        elif ext in (".txt",):
            docs.append(load_text(p))
        else:
            continue
    return docs