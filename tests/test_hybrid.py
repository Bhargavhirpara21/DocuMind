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


class FakeBm25Retriever:
    def __init__(self, nodes, results):
        self.nodes = nodes
        self._results = results

    def retrieve(self, query: str, top_k: int = 5):
        return self._results[:top_k]


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


def test_hybrid_retriever_prioritizes_cover_pages_for_cover_questions() -> None:
    page_one = TextNode(
        text="Product highlights",
        id_="page-1",
        metadata={"document": "cover.pdf", "page": 1},
    )
    page_late = TextNode(
        text="Long technical page content",
        id_="page-136",
        metadata={"document": "cover.pdf", "page": 136},
    )

    hybrid = HybridRetriever(
        FakeVectorIndex([NodeWithScore(node=page_late, score=1.0)]),
        FakeBm25Retriever(
            nodes=[page_one, page_late],
            results=[NodeWithScore(node=page_late, score=1.0)],
        ),
        top_k=2,
    )

    results = hybrid.retrieve("What title is shown on the cover?")
    assert [item.node.metadata["page"] for item in results][:2] == [1, 136]


def test_hybrid_retriever_prioritizes_explicit_page_hints() -> None:
    page_three = TextNode(
        text="A1 - ISO turning",
        id_="page-3",
        metadata={"document": "toc.pdf", "page": 3},
    )
    page_late = TextNode(
        text="Later technical content",
        id_="page-242",
        metadata={"document": "toc.pdf", "page": 242},
    )

    hybrid = HybridRetriever(
        FakeVectorIndex([NodeWithScore(node=page_late, score=1.0)]),
        FakeBm25Retriever(
            nodes=[page_three, page_late],
            results=[NodeWithScore(node=page_late, score=1.0)],
        ),
        top_k=2,
    )

    results = hybrid.retrieve("What product system is listed on page 3 under A2 - Stechen?")
    assert [item.node.metadata["page"] for item in results][:2] == [3, 242]
