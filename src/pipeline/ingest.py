from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src import config
from src.embeddings.embedder import get_embedding_model
from src.ingestion.loader import chunk_documents, load_pdfs
from src.retrieval.bm25 import build_bm25_index, save_bm25_index
from src.retrieval.vector_store import build_index


def _reset_index_storage() -> None:
    for path in (config.CHROMA_DIR, config.BM25_DIR):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def run_ingest(pdf_dir: Path, reset_indexes: bool = False) -> dict[str, int]:
    return run_ingest_with_limit(pdf_dir, reset_indexes=reset_indexes)


def run_ingest_with_limit(
    pdf_dir: Path,
    max_documents: int | None = None,
    reset_indexes: bool = False,
) -> dict[str, int]:
    if reset_indexes:
        _reset_index_storage()

    config.ensure_dirs()
    documents = load_pdfs(pdf_dir)
    if not documents:
        return {"pdfs": 0, "chunks": 0, "vectors": 0}

    if max_documents is not None:
        documents = documents[:max_documents]

    nodes = chunk_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    embed_model = get_embedding_model()

    build_index(nodes, embed_model)
    build_bm25_index(nodes)
    save_bm25_index(nodes, config.BM25_PATH)

    pdf_names = {doc.metadata.get("document") for doc in documents if doc.metadata}
    pdf_count = len({name for name in pdf_names if name})

    return {"pdfs": pdf_count, "chunks": len(nodes), "vectors": len(nodes)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocuMind ingestion pipeline.")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=config.PDF_DIR,
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional limit for smoke testing a subset of loaded PDF documents.",
    )
    parser.add_argument(
        "--reset-indexes",
        action="store_true",
        help="Remove existing ChromaDB and BM25 indexes before ingesting.",
    )
    args = parser.parse_args()

    stats = run_ingest_with_limit(
        args.pdf_dir,
        max_documents=args.max_documents,
        reset_indexes=args.reset_indexes,
    )
    if stats["pdfs"] == 0:
        print("No PDFs found. Add files to data/pdfs and re-run.")
        return 1

    print(f"PDFs loaded: {stats['pdfs']}")
    print(f"Chunks created: {stats['chunks']}")
    print(f"Vectors stored: {stats['vectors']}")
    print(f"BM25 index saved: {config.BM25_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
