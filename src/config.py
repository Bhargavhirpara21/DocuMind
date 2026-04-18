from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_DIR = DATA_DIR / "bm25"
BM25_PATH = BM25_DIR / "index.json"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-de"
)
TOP_K = int(os.getenv("TOP_K", "5"))

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documind")


def ensure_dirs() -> None:
    for path in (PDF_DIR, CHROMA_DIR, BM25_DIR):
        path.mkdir(parents=True, exist_ok=True)


def as_dict() -> dict[str, str]:
    return {
        "ROOT_DIR": str(ROOT_DIR),
        "PDF_DIR": str(PDF_DIR),
        "CHROMA_DIR": str(CHROMA_DIR),
        "BM25_PATH": str(BM25_PATH),
        "CHUNK_SIZE": str(CHUNK_SIZE),
        "CHUNK_OVERLAP": str(CHUNK_OVERLAP),
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "TOP_K": str(TOP_K),
        "LLM_PROVIDER": LLM_PROVIDER,
        "OLLAMA_MODEL": OLLAMA_MODEL,
        "GEMINI_API_KEY_SET": str(bool(GEMINI_API_KEY)),
        "CHROMA_COLLECTION": CHROMA_COLLECTION,
    }


if __name__ == "__main__":
    ensure_dirs()
    for key, value in as_dict().items():
        print(f"{key}={value}")
