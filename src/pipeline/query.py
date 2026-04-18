from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.embeddings.embedder import get_embedding_model
from src.generation.llm import get_llm
from src.generation.prompt import format_prompt
from src.retrieval.bm25 import load_bm25_index
from src.retrieval.hybrid import get_hybrid_retriever
from src.retrieval.vector_store import load_index


@dataclass
class QueryEngine:
    retriever: object
    llm: object


def _unwrap_node(item):
    return getattr(item, "node", item)


def _format_context(nodes) -> str:
    blocks = []
    for node in nodes:
        source = _unwrap_node(node)
        meta = source.metadata or {}
        document = meta.get("document", "unknown")
        page = meta.get("page", "na")
        blocks.append(f"[{document} | page {page}]\n{source.text}")
    return "\n\n".join(blocks)


def _extract_sources(nodes) -> list[dict]:
    sources = []
    seen = set()
    for node in nodes:
        source = _unwrap_node(node)
        meta = source.metadata or {}
        chunk_id = meta.get("chunk_id") or getattr(source, "node_id", None)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        sources.append(
            {
                "document": meta.get("document", "unknown"),
                "page": meta.get("page", "na"),
                "text": source.text,
            }
        )
    return sources


def setup_query_engine() -> QueryEngine:
    embed_model = get_embedding_model()
    vector_index = load_index(embed_model)

    if not Path(config.BM25_PATH).exists():
        raise FileNotFoundError("BM25 index not found. Run ingestion first.")
    bm25_retriever = load_bm25_index(config.BM25_PATH)

    retriever = get_hybrid_retriever(vector_index, bm25_retriever, config.TOP_K)
    llm = get_llm()

    return QueryEngine(retriever=retriever, llm=llm)


def ask(question: str, engine: QueryEngine | None = None) -> dict:
    engine = engine or setup_query_engine()
    nodes = engine.retriever.retrieve(question)

    if not nodes:
        return {
            "answer": "I cannot find this information in the available documents.",
            "sources": [],
        }

    context = _format_context(nodes)
    prompt = format_prompt(context=context, question=question)
    response = engine.llm.complete(prompt)
    answer = getattr(response, "text", str(response))

    return {"answer": answer, "sources": _extract_sources(nodes)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocuMind query pipeline.")
    parser.add_argument("question", type=str, help="Question to ask the system.")
    args = parser.parse_args()

    result = ask(args.question)
    print(result["answer"])
    if result["sources"]:
        print("Sources:")
        for source in result["sources"]:
            print(f"- {source['document']} (page {source['page']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
