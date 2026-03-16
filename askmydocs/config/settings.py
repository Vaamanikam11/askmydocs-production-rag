from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val.strip() != "" else default


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_raw: Path
    data_processed: Path
    chroma_dir: Path
    bm25_dir: Path


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 700       # within 500–800 recommended range
    overlap_tokens: int = 100
    min_tokens: int = 120          # drop very small chunks
    tokenizer_name: str = "o200k_base"  # tiktoken encoder (stable)


@dataclass(frozen=True)
class IngestConfig:
    allowed_exts: tuple[str, ...] = (".pdf", ".md", ".markdown", ".txt")
    max_files: int | None = None   # set to an int for quick tests


def get_paths() -> Paths:
    repo_root = Path(__file__).resolve().parents[2]
    data_raw = repo_root / "data" / "raw"
    data_processed = repo_root / "data" / "processed"
    chroma_dir = repo_root / "data" / "index" / "chroma"
    bm25_dir = repo_root / "data" / "index" / "bm25"

    data_raw.mkdir(parents=True, exist_ok=True)
    data_processed.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    bm25_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        repo_root=repo_root,
        data_raw=data_raw,
        data_processed=data_processed,
        chroma_dir=chroma_dir,
        bm25_dir=bm25_dir,
    )


def chunking_config_from_env() -> ChunkingConfig:
    # Keep env optional; defaults match production-reasonable settings
    target = int(_env("CHUNK_TOKENS", "700"))
    overlap = int(_env("CHUNK_OVERLAP_TOKENS", "100"))
    min_tokens = int(_env("CHUNK_MIN_TOKENS", "120"))
    tokenizer_name = _env("TIKTOKEN_ENCODING", "o200k_base")
    return ChunkingConfig(
        target_tokens=target,
        overlap_tokens=overlap,
        min_tokens=min_tokens,
        tokenizer_name=tokenizer_name,
    )