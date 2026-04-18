from types import SimpleNamespace

from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


class FakeRetriever:
    def __init__(self, nodes):
        self._nodes = nodes

    def retrieve(self, query: str):
        return self._nodes


class FakeVectorIndex:
    def __init__(self, nodes):
        self._nodes = nodes

    def as_retriever(self, similarity_top_k: int):
        return FakeRetriever(self._nodes)


def test_reciprocal_rank_fusion_prefers_top_matches() -> None:
    node_a = TextNode(text="A", id_="a", metadata={"document": "a.pdf", "page": 1})
    node_b = TextNode(text="B", id_="b", metadata={"document": "b.pdf", "page": 2})
    node_c = TextNode(text="C", id_="c", metadata={"document": "c.pdf", "page": 3})

    results = reciprocal_rank_fusion(
        [
            [NodeWithScore(node=node_a, score=1.0), NodeWithScore(node=node_b, score=0.8)],
            [NodeWithScore(node=node_b, score=1.0), NodeWithScore(node=node_c, score=0.7)],
        ],
        top_k=2,
    )

    assert [item.node.text for item in results] == ["B", "A"]


def test_hybrid_retriever_combines_sources() -> None:
    node_a = TextNode(text="A", id_="a", metadata={"document": "a.pdf", "page": 1})
    node_b = TextNode(text="B", id_="b", metadata={"document": "b.pdf", "page": 2})

    hybrid = HybridRetriever(
        FakeVectorIndex([NodeWithScore(node=node_a, score=1.0)]),
        FakeRetriever([NodeWithScore(node=node_b, score=1.0)]),
        top_k=2,
    )

    results = hybrid.retrieve("question")
    assert [item.node.text for item in results] == ["A", "B"]
