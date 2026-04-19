from types import SimpleNamespace

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


class FakeEngine:
    retriever = FakeRetriever()
    llm = FakeLLM()


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
