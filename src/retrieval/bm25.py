from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from llama_index.core.schema import NodeWithScore, TextNode
from rank_bm25 import BM25Okapi


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _serialize_node(node: TextNode) -> dict[str, Any]:
    return {
        "id": node.node_id,
        "text": node.text,
        "metadata": {
            key: _json_safe(value) for key, value in (node.metadata or {}).items()
        },
    }


def _deserialize_node(data: dict[str, Any]) -> TextNode:
    return TextNode(
        text=data.get("text", ""),
        id_=data.get("id"),
        metadata=data.get("metadata", {}),
    )


def load_bm25_nodes(path: Path) -> list[TextNode]:
    path = Path(path)
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [_deserialize_node(item) for item in payload]


def merge_bm25_nodes(existing: list[TextNode], new: list[TextNode]) -> list[TextNode]:
    merged = {node.node_id: node for node in existing}
    for node in new:
        merged[node.node_id] = node
    return list(merged.values())


def _default_tokenizer(text: str) -> list[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(
        self, nodes: list[TextNode], tokenizer: Callable[[str], list[str]] | None = None
    ) -> None:
        self._nodes = nodes
        self._tokenizer = tokenizer or _default_tokenizer
        self._corpus = [self._tokenizer(node.text) for node in nodes]
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def retrieve(self, query: str, top_k: int = 5) -> list[NodeWithScore]:
        if not self._bm25 or not query:
            return []
        tokens = self._tokenizer(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        results = []
        for idx in ranked[:top_k]:
            node = self._nodes[idx]
            results.append(NodeWithScore(node=node, score=float(scores[idx])))
        return results


def build_bm25_index(nodes: list[TextNode]) -> BM25Retriever:
    return BM25Retriever(nodes=nodes)


def save_bm25_index(nodes: list[TextNode], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_serialize_node(node) for node in nodes]
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def load_bm25_index(path: Path) -> BM25Retriever:
    nodes = load_bm25_nodes(path)
    return build_bm25_index(nodes)
