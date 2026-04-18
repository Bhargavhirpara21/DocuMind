from types import SimpleNamespace
from pathlib import Path

from src.pipeline.ingest import run_ingest_with_limit


def test_run_ingest_with_limit_truncates_loaded_documents(monkeypatch, tmp_path) -> None:
    documents = [
        SimpleNamespace(metadata={"document": "a.pdf"}),
        SimpleNamespace(metadata={"document": "b.pdf"}),
        SimpleNamespace(metadata={"document": "c.pdf"}),
    ]
    chunks = [SimpleNamespace(metadata={"document": "a.pdf"})]

    monkeypatch.setattr("src.pipeline.ingest.config.ensure_dirs", lambda: None)
    monkeypatch.setattr("src.pipeline.ingest.load_pdfs", lambda pdf_dir: documents)
    monkeypatch.setattr(
        "src.pipeline.ingest.chunk_documents", lambda docs, chunk_size, chunk_overlap: chunks
    )
    monkeypatch.setattr("src.pipeline.ingest.get_embedding_model", lambda: object())
    monkeypatch.setattr("src.pipeline.ingest.build_index", lambda chunks, embed_model: None)
    monkeypatch.setattr("src.pipeline.ingest.build_bm25_index", lambda chunks: None)
    monkeypatch.setattr("src.pipeline.ingest.save_bm25_index", lambda nodes, path: None)

    stats = run_ingest_with_limit(Path(tmp_path), max_documents=1)

    assert stats == {"pdfs": 1, "chunks": 1, "vectors": 1}
