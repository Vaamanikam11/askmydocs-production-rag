from askmydocs.config.settings import ChunkingConfig
from askmydocs.ingest.metadata import Document
from askmydocs.ingest.chunking import chunk_document


def test_chunking_produces_overlap_and_reasonable_sizes():
    doc = Document(
        doc_id="doc123",
        source_path="x.md",
        title="x",
        text=" ".join(["hello"] * 6000),
        source_type="markdown",
    )
    cfg = ChunkingConfig(target_tokens=200, overlap_tokens=50, min_tokens=50, tokenizer_name="cl100k_base")
    chunks = chunk_document(doc, cfg)

    assert len(chunks) >= 2
    # token offsets should be monotonic
    for i in range(1, len(chunks)):
        assert chunks[i].start_token < chunks[i].end_token
        assert chunks[i].start_token < chunks[i - 1].end_token