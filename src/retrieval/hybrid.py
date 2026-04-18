from __future__ import annotations

from llama_index.core.schema import NodeWithScore


def _get_node(item) -> object:
    return getattr(item, "node", item)


def _get_node_id(item) -> str:
    node = _get_node(item)
    return (
        getattr(node, "node_id", None)
        or getattr(node, "id_", None)
        or getattr(node, "id", None)
        or str(id(node))
    )


def reciprocal_rank_fusion(
    result_lists: list[list[object]], top_k: int, k: int = 60
) -> list[NodeWithScore]:
    fused_scores: dict[str, float] = {}
    nodes_by_id: dict[str, object] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            node_id = _get_node_id(item)
            nodes_by_id[node_id] = _get_node(item)
            fused_scores[node_id] = fused_scores.get(node_id, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [
        NodeWithScore(node=nodes_by_id[node_id], score=fused_scores[node_id])
        for node_id in ranked_ids[:top_k]
    ]


class HybridRetriever:
    def __init__(self, vector_index, bm25_retriever, top_k: int) -> None:
        self._vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)
        self._bm25_retriever = bm25_retriever
        self._top_k = top_k

    def retrieve(self, query: str) -> list[NodeWithScore]:
        vector_results = self._vector_retriever.retrieve(query)
        try:
            bm25_results = self._bm25_retriever.retrieve(query, top_k=self._top_k)
        except TypeError:
            bm25_results = self._bm25_retriever.retrieve(query)
        return reciprocal_rank_fusion(
            [vector_results, bm25_results], top_k=self._top_k
        )


def get_hybrid_retriever(vector_index, bm25_retriever, top_k: int) -> HybridRetriever:
    return HybridRetriever(vector_index, bm25_retriever, top_k=top_k)


def retrieve(query: str, retriever: HybridRetriever):
    return retriever.retrieve(query)
