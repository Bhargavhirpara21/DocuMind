from __future__ import annotations

import gc
import argparse
import shutil
from typing import Iterable
from pathlib import Path

from llama_index.core.schema import Document

from src import config
from src.embeddings.embedder import get_embedding_model
from src.ingestion.loader import chunk_documents, load_pdf_paths, load_pdfs
from src.retrieval.bm25 import (
    build_bm25_index,
    load_bm25_nodes,
    merge_bm25_nodes,
    save_bm25_index,
)
from src.retrieval.vector_store import build_index, get_vector_store


def _reset_index_storage() -> None:
    for path in (config.CHROMA_DIR, config.BM25_DIR):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def _delete_existing_documents(documents: list[Document]) -> None:
    vector_store = get_vector_store()
    document_ids: list[str] = []
    for document in documents:
        document_id = getattr(document, "id_", None)
        if document_id and document_id not in document_ids:
            document_ids.append(document_id)

    for document_id in document_ids:
        vector_store.delete(document_id)


def _ingest_documents(
    documents: list[Document],
    reset_indexes: bool = False,
    merge_bm25: bool = False,
    replace_existing_documents: bool = False,
) -> dict[str, int]:
    if reset_indexes:
        print("Resetting existing indexes...", flush=True)
        _reset_index_storage()

    config.ensure_dirs()
    if not documents:
        return {"pdfs": 0, "chunks": 0, "vectors": 0}

    if replace_existing_documents:
        print("Replacing existing documents in Chroma...", flush=True)
        _delete_existing_documents(documents)

    print(f"Loaded {len(documents)} PDF documents.", flush=True)

    print("Chunking documents...", flush=True)
    nodes = chunk_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"Created {len(nodes)} chunks.", flush=True)

    print("Loading embedding model...", flush=True)
    embed_model = get_embedding_model()

    print("Building Chroma index...", flush=True)
    build_index(nodes, embed_model)
    print("Building BM25 index...", flush=True)

    bm25_nodes = nodes
    if merge_bm25:
        existing_nodes = load_bm25_nodes(config.BM25_PATH)
        bm25_nodes = merge_bm25_nodes(existing_nodes, nodes)

    build_bm25_index(bm25_nodes)
    print("Saving BM25 index...", flush=True)
    save_bm25_index(bm25_nodes, config.BM25_PATH)

    gc.collect()

    pdf_names = {doc.metadata.get("document") for doc in documents if doc.metadata}
    pdf_count = len({name for name in pdf_names if name})

    return {"pdfs": pdf_count, "chunks": len(nodes), "vectors": len(nodes)}


def run_ingest(pdf_dir: Path, reset_indexes: bool = False) -> dict[str, int]:
    return run_ingest_with_limit(pdf_dir, reset_indexes=reset_indexes)


def run_ingest_with_limit(
    pdf_dir: Path,
    max_documents: int | None = None,
    reset_indexes: bool = False,
) -> dict[str, int]:
    print(f"Loading PDFs from {pdf_dir}...", flush=True)
    documents = load_pdfs(pdf_dir)
    if not documents:
        return {"pdfs": 0, "chunks": 0, "vectors": 0}

    if max_documents is not None:
        documents = documents[:max_documents]

    return _ingest_documents(documents, reset_indexes=reset_indexes)


def run_upload_ingest(
    pdf_paths: Iterable[Path],
    reset_indexes: bool = False,
) -> dict[str, int]:
    print("Loading uploaded PDFs...", flush=True)
    documents = load_pdf_paths(pdf_paths)
    return _ingest_documents(
        documents,
        reset_indexes=reset_indexes,
        merge_bm25=True,
        replace_existing_documents=True,
    )


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
