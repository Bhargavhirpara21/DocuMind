from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.query import ask, setup_query_engine


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_evaluation(questions: list[dict]) -> list[dict]:
    if not questions:
        return []

    engine = setup_query_engine()
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
