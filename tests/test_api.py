from fastapi.testclient import TestClient

from src.api.routes import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_documents_endpoint_returns_list() -> None:
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_upload_endpoint_ingests_only_uploaded_file(monkeypatch, tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    captured: dict[str, list] = {}

    monkeypatch.setattr("src.api.routes.config.PDF_DIR", pdf_dir)
    monkeypatch.setattr("src.api.routes.config.ensure_dirs", lambda: pdf_dir.mkdir(parents=True, exist_ok=True))

    def fake_run_upload_ingest(pdf_paths):
        captured["pdf_paths"] = list(pdf_paths)
        return {"pdfs": 1, "chunks": 3, "vectors": 3}

    monkeypatch.setattr("src.api.routes.run_upload_ingest", fake_run_upload_ingest)

    response = client.post(
        "/api/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "uploaded"
    assert response.json()["filename"] == "sample.pdf"
    assert response.json()["chunks"] == 3
    assert captured["pdf_paths"] == [pdf_dir / "sample.pdf"]
