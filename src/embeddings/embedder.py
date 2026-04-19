from __future__ import annotations

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except Exception:
    from llama_index.core.embeddings import HuggingFaceEmbedding

from src import config

try:
    import torch
except Exception:
    torch = None


def _default_device() -> str | None:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return None


def get_embedding_model(model_name: str | None = None) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(
        model_name=model_name or config.EMBEDDING_MODEL,
        device=_default_device(),
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    embed_model = get_embedding_model()
    return embed_model.get_text_embedding_batch(texts)


if __name__ == "__main__":
    vectors = embed_texts(["DocuMind embedding sanity check."])
    print(len(vectors[0]))
