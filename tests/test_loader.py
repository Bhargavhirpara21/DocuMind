from pathlib import Path

import pytest

pytest.importorskip("llama_index")

from llama_index.core.schema import Document

from src.ingestion import loader
from src.ingestion.loader import chunk_documents


def test_chunk_documents_sets_metadata_and_id() -> None:
    docs = [
        Document(
            text="Hello world",
            id_="file:///tmp/file.pdf",
            metadata={"document": "file.pdf", "page": 1},
        )
    ]
    nodes = chunk_documents(docs, chunk_size=20, chunk_overlap=0)
    assert nodes
    node = nodes[0]
    assert node.metadata["document"] == "file.pdf"
    assert node.metadata["page"] == 1
    assert node.metadata["chunk_id"] == node.node_id
    assert node.ref_doc_id == docs[0].id_


def test_load_pdf_assigns_unique_ids_per_page(monkeypatch) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self):
            return self._text

    class FakePdfReader:
        def __init__(self, path) -> None:
            self.pages = [FakePage("Page one"), FakePage("Page two")]

    monkeypatch.setattr(loader, "PdfReader", FakePdfReader)

    docs = loader._load_pdf(Path("sample.pdf"), index=1, total_files=1)

    assert [doc.metadata["page"] for doc in docs] == [1, 2]
    assert docs[0].id_ != docs[1].id_
    assert docs[0].id_.endswith("::page-1")
    assert docs[1].id_.endswith("::page-2")
