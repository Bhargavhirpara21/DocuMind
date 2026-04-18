from src.pipeline.ingest import run_ingest


def test_run_ingest_handles_empty_directory(tmp_path) -> None:
    stats = run_ingest(tmp_path)
    assert stats == {"pdfs": 0, "chunks": 0, "vectors": 0}
