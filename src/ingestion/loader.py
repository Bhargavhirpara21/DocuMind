from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

try:
    from llama_index.readers.file import PDFReader
except Exception:
    from llama_index.core.readers.file import PDFReader


def _safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _coerce_page_number(value: object) -> int | None:
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return int(stripped)

        match = re.search(r"(\d+)", stripped)
        if match:
            return int(match.group(1))

    return _safe_int(value)


def _normalize_metadata(metadata: dict | None) -> dict:
    meta = dict(metadata or {})
    file_name = (
        meta.get("file_name")
        or meta.get("filename")
        or meta.get("source")
        or meta.get("document")
    )
    if file_name:
        meta["document"] = str(file_name)
    else:
        meta["document"] = "unknown"

    page_value = meta.get("page_number") or meta.get("page_label") or meta.get("page")
    page_int = _coerce_page_number(page_value)
    if page_int is not None:
        meta["page"] = page_int

    return meta


def load_pdfs(pdf_dir: Path) -> list[Document]:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        return []

    pdf_paths = sorted(p for p in pdf_dir.glob("*.pdf") if p.is_file())
    if not pdf_paths:
        return []

    reader = SimpleDirectoryReader(
        input_files=[str(path) for path in pdf_paths],
        file_extractor={".pdf": PDFReader()},
    )
    documents = reader.load_data()

    for doc in documents:
        doc.metadata = _normalize_metadata(doc.metadata)

    return documents


def chunk_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(list(documents))

    for index, node in enumerate(nodes):
        node.metadata = _normalize_metadata(node.metadata)
        doc_name = Path(str(node.metadata.get("document", "unknown"))).name
        page = node.metadata.get("page", "na")
        chunk_id = f"{doc_name}|{page}|{index}"
        node.metadata["chunk_id"] = chunk_id
        node.id_ = chunk_id

    return nodes


if __name__ == "__main__":
    from src import config

    docs = load_pdfs(config.PDF_DIR)
    nodes = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"PDFs loaded: {len(docs)}")
    print(f"Chunks created: {len(nodes)}")
