from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Citation:
    cite_key: str   # e.g., [1]
    chunk_id: str
    source_path: str
    title: str
    page: str
    preview: str


def format_citation_block(citations: list[Citation]) -> str:
    lines = ["\nSources:"]
    for c in citations:
        src = Path(c.source_path).name
        page = ""
        if c.page is not None and str(c.page).strip() != "" and str(c.page).lower() != "none":
            page = f", p.{c.page}"
        lines.append(f"{c.cite_key} {src}{page} — {c.title} (chunk: {c.chunk_id})")
    return "\n".join(lines)


def build_citations(hits: list, max_sources: int = 5) -> tuple[list[Citation], dict[str, str]]:
    """
    Returns:
      - list of Citation objects
      - map chunk_id -> cite_key (e.g., {"doc::c00001": "[1]"})
    """
    citations: list[Citation] = []
    chunk_to_key: dict[str, str] = {}

    for i, h in enumerate(hits[:max_sources], start=1):
        key = f"[{i}]"
        chunk_to_key[h.chunk_id] = key

        meta = h.meta or {}
        source_path = str(meta.get("source_path", ""))
        title = str(meta.get("title", ""))
        page = str(meta.get("page", ""))

        preview = (h.text or "").strip().replace("\n", " ")
        preview = preview[:220] + ("..." if len(preview) > 220 else "")

        citations.append(
            Citation(
                cite_key=key,
                chunk_id=h.chunk_id,
                source_path=source_path,
                title=title,
                page=page,
                preview=preview,
            )
        )

    return citations, chunk_to_key