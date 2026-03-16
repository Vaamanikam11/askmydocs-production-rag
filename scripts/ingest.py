from __future__ import annotations

from pathlib import Path
import json
import argparse
import askmydocs.config.env 

from askmydocs.config.settings import get_paths, chunking_config_from_env, IngestConfig
from askmydocs.ingest.loaders import load_documents
from askmydocs.ingest.chunking import chunk_documents


def iter_input_files(root: Path, allowed_exts: tuple[str, ...], max_files: int | None):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed_exts:
            files.append(p)
    files.sort()
    if max_files is not None:
        files = files[:max_files]
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Input folder (default: data/raw)")
    parser.add_argument("--out", type=str, default=None, help="Output JSONL (default: data/processed/chunks.jsonl)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files for quick testing")
    args = parser.parse_args()

    paths = get_paths()
    ingest_cfg = IngestConfig(max_files=args.max_files)

    input_dir = Path(args.input) if args.input else paths.data_raw
    out_path = Path(args.out) if args.out else (paths.data_processed / "chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(input_dir, ingest_cfg.allowed_exts, ingest_cfg.max_files)
    if not files:
        print(f"No input files found in: {input_dir}")
        return

    docs = load_documents(files)
    chunk_cfg = chunking_config_from_env()
    chunks = chunk_documents(docs, chunk_cfg)

    # Write chunks JSONL for downstream indexing (vector + bm25)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            row = {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "source_path": c.source_path,
                "source_type": c.source_type,
                "page": c.page,
                "start_token": c.start_token,
                "end_token": c.end_token,
                "title": c.title,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Ingested files: {len(files)}")
    print(f"Documents: {len(docs)}")
    print(f"Chunks written: {len(chunks)} -> {out_path}")

        # Build indexes
    from askmydocs.index.chroma_store import ChromaIndex
    from askmydocs.index.bm25_store import BM25Index

    # Vector index (Chroma)
    chroma = ChromaIndex(
        persist_dir=str(paths.chroma_dir),
        collection_name="askmydocs",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    chroma.reset()
    chroma.add_chunks([json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()])

    # BM25 index
    bm25 = BM25Index(str(paths.bm25_dir))
    bm25.build([json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()])
    bm25.save()

    print("Indexes built: Chroma + BM25")


if __name__ == "__main__":
    main()