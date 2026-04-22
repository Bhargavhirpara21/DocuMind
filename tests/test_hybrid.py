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


def test_hybrid_retriever_prioritizes_cover_pages_for_application_area_questions() -> None:
    page_one = TextNode(
        text="Turning, Holemaking, Threading Special Edition Edition 2026 Information and Order Data Small Part Machining Aerospace",
        id_="page-1",
        metadata={"document": "flyer.pdf", "page": 1},
    )
    page_late = TextNode(
        text="Later technical content about tooling",
        id_="page-132",
        metadata={"document": "flyer.pdf", "page": 132},
    )

    hybrid = HybridRetriever(
        FakeVectorIndex([NodeWithScore(node=page_late, score=1.0)]),
        FakeBm25Retriever(
            nodes=[page_one, page_late],
            results=[NodeWithScore(node=page_late, score=1.0)],
        ),
        top_k=2,
    )

    results = hybrid.retrieve("What application area is the flyer for?")
    assert [item.node.metadata["page"] for item in results][:2] == [1, 132]


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


def test_hybrid_retriever_prioritizes_page_three_for_iso_turning_summary_questions() -> None:
    page_three = TextNode(
        text=(
            "3 ISO turning Page Tiger·tec® Gold turning grades WMP20G, WMP30G 4 RM7 roughing geometry "
            "for ISO M 6 Walter Turn long turn holder P.-S-P 7 WB solid carbide mini boring bars and adaptors 8"
        ),
        id_="page-3",
        metadata={"document": "highlights.pdf", "page": 3},
    )
    page_five = TextNode(
        text="5 Highest repeat accuracy and stability during copy turning thanks to form-fitting WL17 cutting inserts",
        id_="page-5",
        metadata={"document": "highlights.pdf", "page": 5},
    )

    hybrid = HybridRetriever(
        FakeVectorIndex([NodeWithScore(node=page_five, score=1.0)]),
        FakeBm25Retriever(
            nodes=[page_three, page_five],
            results=[NodeWithScore(node=page_five, score=1.0)],
        ),
        top_k=2,
    )

    results = hybrid.retrieve("What product grades are listed first under ISO turning?")
    assert [item.node.metadata["page"] for item in results][:2] == [3, 5]


def test_hybrid_retriever_prioritizes_page_three_for_groovtec_summary_questions() -> None:
    page_three = TextNode(
        text=(
            "3 ISO turning Page Tiger·tec® Gold turning grades WMP20G, WMP30G 10 Grooving Page "
            "Groov·tec® GD grooving system G5011 12 Groov·tec® GD axial grooving system G5111"
        ),
        id_="page-3",
        metadata={"document": "highlights.pdf", "page": 3},
    )
    page_five = TextNode(
        text="5 Highest repeat accuracy and stability during copy turning thanks to form-fitting WL17 cutting inserts",
        id_="page-5",
        metadata={"document": "highlights.pdf", "page": 5},
    )

    hybrid = HybridRetriever(
        FakeVectorIndex([NodeWithScore(node=page_five, score=1.0)]),
        FakeBm25Retriever(
            nodes=[page_three, page_five],
            results=[NodeWithScore(node=page_five, score=1.0)],
        ),
        top_k=2,
    )

    results = hybrid.retrieve("What Groovtec item is listed under Grooving?")
    assert [item.node.metadata["page"] for item in results][:2] == [3, 5]
