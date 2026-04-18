from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.embeddings.embedder import get_embedding_model
from src.ingestion.loader import chunk_documents, load_pdfs
from src.retrieval.bm25 import build_bm25_index, save_bm25_index
from src.retrieval.vector_store import build_index


def run_ingest(pdf_dir: Path) -> dict[str, int]:
    config.ensure_dirs()
    documents = load_pdfs(pdf_dir)
    if not documents:
        return {"pdfs": 0, "chunks": 0, "vectors": 0}

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
    args = parser.parse_args()

    stats = run_ingest(args.pdf_dir)
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
