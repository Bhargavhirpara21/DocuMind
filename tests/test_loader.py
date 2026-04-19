import pytest

pytest.importorskip("llama_index")

from llama_index.core.schema import Document

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
