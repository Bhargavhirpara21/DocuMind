from __future__ import annotations

from src import config


def _import_gemini():
    try:
        from llama_index.llms.gemini import Gemini
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Gemini LLM is not installed. Add llama-index-llms-gemini."
        ) from exc
    return Gemini


def _import_ollama():
    try:
        from llama_index.llms.ollama import Ollama
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Ollama LLM is not installed. Add llama-index-llms-ollama."
        ) from exc
    return Ollama


def get_llm():
    provider = config.LLM_PROVIDER
    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        Gemini = _import_gemini()
        return Gemini(model="models/gemini-2.5-flash", api_key=config.GEMINI_API_KEY)

    if provider == "ollama":
        Ollama = _import_ollama()
        return Ollama(model=config.OLLAMA_MODEL)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


if __name__ == "__main__":
    llm = get_llm()
    response = llm.complete("What is 2+2?")
    print(getattr(response, "text", str(response)))
