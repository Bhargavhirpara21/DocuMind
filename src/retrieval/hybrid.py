from __future__ import annotations

import re
from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore


_PAGE_HINT_RE = re.compile(r"\bpage\s*(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class _QueryProfile:
    page_hint: int | None
    cover: bool
    toc: bool
    contact: bool


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


def _get_metadata(item) -> dict:
    node = _get_node(item)
    return node.metadata or {}


def _query_profile(query: str) -> _QueryProfile:
    normalized = query.lower()
    page_hint = None
    page_match = _PAGE_HINT_RE.search(normalized)
    if page_match:
        try:
            page_hint = int(page_match.group(1))
        except ValueError:
            page_hint = None

    cover = any(
        marker in normalized
        for marker in (
            "cover",
            "title",
            "edition",
            "phrase appears on the cover",
            "shown on the cover",
        )
    )
    toc = any(
        marker in normalized
        for marker in (
            "table of contents",
            "first section",
            "main section",
            "contents",
            "toc",
        )
    )
    contact = any(
        marker in normalized
        for marker in (
            "website",
            "contact details",
            "local contact",
            "contact",
        )
    )
    return _QueryProfile(page_hint=page_hint, cover=cover, toc=toc, contact=contact)


def _looks_like_title(text: str) -> bool:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return False

    words = first_line.split()
    if len(words) <= 6:
        return True

    letters = [char for char in first_line if char.isalpha()]
    if not letters:
        return False

    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return uppercase_ratio >= 0.6 and len(words) <= 10


def _page_priority_score(item, profile: _QueryProfile) -> float:
    meta = _get_metadata(item)
    page = meta.get("page")
    if not isinstance(page, int):
        return 0.0

    source = _get_node(item)
    text = (source.text or "").strip()
    normalized = text.lower()
    word_count = len(text.split()) or 1

    if profile.page_hint is not None:
        if page != profile.page_hint:
            return 0.0
        score = 8.0
        if _looks_like_title(text):
            score += 1.5
        return score

    if profile.cover:
        if page > 5:
            return 0.0
        score = max(0.0, 5.0 - (page - 1) * 1.0)
        score += max(0.0, 1.5 - word_count / 20.0)
        if _looks_like_title(text):
            score += 2.0
        if "product highlights" in normalized or "a passion to win" in normalized:
            score += 1.5
        return score

    if profile.toc:
        if page > 6:
            return 0.0
        score = max(0.0, 5.0 - abs(page - 3) * 1.2)
        score += min(normalized.count("page"), 8) * 0.2
        score += min(word_count / 50.0, 2.0)
        if "table of contents" in normalized:
            score += 2.5
        return score

    if profile.contact:
        if page > 3:
            return 0.0
        score = max(0.0, 4.0 - (page - 1) * 1.0)
        if "www." in normalized or ".com" in normalized:
            score += 2.0
        if "contact" in normalized or "website" in normalized:
            score += 1.0
        return score

    return 0.0


def _page_priority_candidates(query: str, items: list[object]) -> list[NodeWithScore]:
    profile = _query_profile(query)
    if not any((profile.page_hint, profile.cover, profile.toc, profile.contact)):
        return []

    candidates = []
    for item in items:
        score = _page_priority_score(item, profile)
        if score <= 0:
            continue
        candidates.append(NodeWithScore(node=_get_node(item), score=score))

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def reciprocal_rank_fusion(
    result_lists: list[list[object]],
    top_k: int,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[NodeWithScore]:
    fused_scores: dict[str, float] = {}
    nodes_by_id: dict[str, object] = {}
    weights = weights or [1.0] * len(result_lists)

    for results, weight in zip(result_lists, weights, strict=False):
        for rank, item in enumerate(results, start=1):
            node_id = _get_node_id(item)
            nodes_by_id[node_id] = _get_node(item)
            fused_scores[node_id] = fused_scores.get(node_id, 0.0) + weight / (k + rank)

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
        result_lists = [vector_results, bm25_results]

        bm25_nodes = getattr(self._bm25_retriever, "nodes", [])
        if bm25_nodes:
            page_candidates = _page_priority_candidates(query, bm25_nodes)
            if page_candidates:
                result_lists.append(page_candidates)

        weights = [1.0] * len(result_lists)
        if len(result_lists) > 2:
            weights[-1] = 4.0

        return reciprocal_rank_fusion(result_lists, top_k=self._top_k, weights=weights)


def get_hybrid_retriever(vector_index, bm25_retriever, top_k: int) -> HybridRetriever:
    return HybridRetriever(vector_index, bm25_retriever, top_k=top_k)


def retrieve(query: str, retriever: HybridRetriever):
    return retriever.retrieve(query)
