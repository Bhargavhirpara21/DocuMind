from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.embeddings.embedder import get_embedding_model
from src.pipeline.query import QueryEngine, ask, setup_query_engine
from src.retrieval.bm25 import load_bm25_index
from src.retrieval.hybrid import get_hybrid_retriever
from src.retrieval.vector_store import load_index


class OfflineLLM:
    def complete(self, prompt: str):
        context_marker = "Context:\n"
        question_marker = "\n\nQuestion:"
        answer = "I cannot find this information in the available documents."

        if context_marker in prompt and question_marker in prompt:
            context = prompt.split(context_marker, 1)[1].split(question_marker, 1)[0].strip()
            first_line = next((line.strip() for line in context.splitlines() if line.strip()), "")
            if first_line:
                answer = first_line

        return type("Response", (), {"text": answer})()


def setup_offline_query_engine() -> QueryEngine:
    embed_model = get_embedding_model()
    vector_index = load_index(embed_model)

    bm25_path = Path(config.BM25_PATH)
    if not bm25_path.exists():
        raise FileNotFoundError("BM25 index not found. Run ingestion first.")

    bm25_retriever = load_bm25_index(bm25_path)
    retriever = get_hybrid_retriever(vector_index, bm25_retriever, 5)
    return QueryEngine(retriever=retriever, llm=OfflineLLM())


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_evaluation(questions: list[dict]) -> list[dict]:
    if not questions:
        return []

    try:
        engine = setup_query_engine()
    except RuntimeError:
        engine = setup_offline_query_engine()
    results = []
    for item in questions:
        question = item.get("question", "").strip()
        if not question:
            continue
        response = ask(question, engine=engine)
        results.append(
            {
                "question": question,
                "answer": response["answer"],
                "sources": response["sources"],
                "expected_answer": item.get("expected_answer"),
                "source_document": item.get("source_document"),
                "source_page": item.get("source_page"),
                "type": item.get("type"),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocuMind evaluation.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).with_name("test_questions.json"),
        help="Path to a JSON file containing evaluation questions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results.json"),
        help="Where to write the collected answers.",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        print("No evaluation questions found. Add questions to evaluation/test_questions.json.")
        return 1

    results = run_evaluation(questions)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote {len(results)} evaluation results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
