from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader

from src import config
from src.pipeline.ingest import run_ingest, run_upload_ingest
from src.pipeline.query import ask as run_ask
from src.pipeline.query import setup_query_engine

app = FastAPI(title="DocuMind API", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


_engine_cache = None


def _reset_engine() -> None:
    global _engine_cache
    _engine_cache = None


def _get_engine():
    global _engine_cache
    if _engine_cache is None:
        _engine_cache = setup_query_engine()
    return _engine_cache


def _count_pages(path: Path) -> int:
    try:
        reader = PdfReader(str(path))
        return len(reader.pages)
    except Exception:
        return 0


def _list_documents(pdf_dir: Path) -> list[dict]:
    if not pdf_dir.exists():
        return []
    items = []
    for path in sorted(pdf_dir.glob("*.pdf")):
        items.append({"name": path.name, "pages": _count_pages(path)})
    return items


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        engine = _get_engine()
        return run_ask(question, engine=engine)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/ingest")
def ingest(pdf_dir: str | None = None) -> dict:
    target = Path(pdf_dir) if pdf_dir else config.PDF_DIR
    stats = run_ingest(target)
    if stats["pdfs"] == 0:
        raise HTTPException(status_code=400, detail="No PDFs found to ingest.")

    _reset_engine()
    return {"status": "ingested", **stats}


@app.get("/api/documents")
def documents() -> list[dict]:
    return _list_documents(config.PDF_DIR)


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    filename = Path(file.filename).name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    config.ensure_dirs()
    dest = config.PDF_DIR / filename
    with dest.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    stats = run_upload_ingest([dest])
    if stats["pdfs"] == 0:
        raise HTTPException(status_code=400, detail="No PDFs found to ingest.")

    _reset_engine()
    return {"status": "uploaded", "filename": filename, "chunks": stats["chunks"]}
