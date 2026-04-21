from types import SimpleNamespace

import pytest
from llama_index.core.schema import TextNode

from src.pipeline.query import ask


class FakeRetriever:
    def retrieve(self, query: str):
        node = TextNode(
            text="Tensile strength is 950 MPa.",
            id_="chunk-1",
            metadata={"document": "sample.pdf", "page": 12, "chunk_id": "chunk-1"},
        )
        return [SimpleNamespace(node=node, score=0.9)]


class FakeLLM:
    def complete(self, prompt: str):
        return SimpleNamespace(text="The tensile strength is 950 MPa.")


class RateLimitedLLM:
    def complete(self, prompt: str):
        raise RuntimeError(
            "429 You exceeded your current quota, please check your plan and billing details."
        )


class BrokenLLM:
    def complete(self, prompt: str):
        raise ValueError("unexpected failure")


class FakeEngine:
    retriever = FakeRetriever()
    llm = FakeLLM()


class RateLimitedEngine:
    retriever = FakeRetriever()
    llm = RateLimitedLLM()


class BrokenEngine:
    retriever = FakeRetriever()
    llm = BrokenLLM()


def test_ask_returns_answer_and_sources() -> None:
    result = ask("What is the tensile strength?", engine=FakeEngine())
    assert result["answer"] == "The tensile strength is 950 MPa."
    assert result["sources"] == [
        {
            "document": "sample.pdf",
            "page": 12,
            "citation": "sample.pdf (page 12)",
            "chunk_id": "chunk-1",
            "text": "Tensile strength is 950 MPa.",
        }
    ]


def test_ask_falls_back_when_llm_is_rate_limited() -> None:
    result = ask("What is the tensile strength?", engine=RateLimitedEngine())

    assert result["answer"] == (
        "The Gemini API quota was exceeded while generating this answer. "
        "Retrieved sources are listed below."
    )
    assert result["sources"] == [
        {
            "document": "sample.pdf",
            "page": 12,
            "citation": "sample.pdf (page 12)",
            "chunk_id": "chunk-1",
            "text": "Tensile strength is 950 MPa.",
        }
    ]


def test_ask_propagates_non_rate_limit_errors() -> None:
    with pytest.raises(ValueError, match="unexpected failure"):
        ask("What is the tensile strength?", engine=BrokenEngine())
