import pytest

pytest.importorskip("llama_index")

from llama_index.core.schema import TextNode

from src.retrieval.bm25 import load_bm25_index, save_bm25_index


def test_bm25_roundtrip(tmp_path) -> None:
    nodes = [TextNode(text="alpha beta", id_="1", metadata={"document": "d", "page": 1})]
    index_path = tmp_path / "bm25.json"
    save_bm25_index(nodes, index_path)

    retriever = load_bm25_index(index_path)
    results = retriever.retrieve("alpha")
    assert results
