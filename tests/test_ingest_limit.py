from pathlib import Path
from types import SimpleNamespace

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


def test_run_ingest_with_limit_can_reset_indexes(monkeypatch, tmp_path) -> None:
    documents = [SimpleNamespace(metadata={"document": "a.pdf"})]
    chunks = [SimpleNamespace(metadata={"document": "a.pdf"})]
    removed_paths = []
    chroma_dir = tmp_path / "chroma"
    bm25_dir = tmp_path / "bm25"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    bm25_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("src.pipeline.ingest.config.ensure_dirs", lambda: None)
    monkeypatch.setattr("src.pipeline.ingest.load_pdfs", lambda pdf_dir: documents)
    monkeypatch.setattr(
        "src.pipeline.ingest.chunk_documents", lambda docs, chunk_size, chunk_overlap: chunks
    )
    monkeypatch.setattr("src.pipeline.ingest.get_embedding_model", lambda: object())
    monkeypatch.setattr("src.pipeline.ingest.build_index", lambda chunks, embed_model: None)
    monkeypatch.setattr("src.pipeline.ingest.build_bm25_index", lambda chunks: None)
    monkeypatch.setattr("src.pipeline.ingest.save_bm25_index", lambda nodes, path: None)
    monkeypatch.setattr(
        "src.pipeline.ingest.shutil.rmtree", lambda path, ignore_errors=True: removed_paths.append(path)
    )
    monkeypatch.setattr("src.pipeline.ingest.config.CHROMA_DIR", chroma_dir)
    monkeypatch.setattr("src.pipeline.ingest.config.BM25_DIR", bm25_dir)
    monkeypatch.setattr("src.pipeline.ingest.config.BM25_PATH", bm25_dir / "index.json")

    stats = run_ingest_with_limit(Path(tmp_path), max_documents=1, reset_indexes=True)

    assert stats == {"pdfs": 1, "chunks": 1, "vectors": 1}
    assert removed_paths
