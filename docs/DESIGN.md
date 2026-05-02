# Design Decisions

## Goals

- Answer questions from manufacturing documents with citations
- Support both exact-code lookups and semantic questions
- Keep the system simple enough to run locally and deploy to Render

## Key Decisions

### LlamaIndex
LlamaIndex is used because the project is fundamentally a RAG system. It reduces boilerplate around document loading, chunking, retrievers, and vector store integration.

### Hybrid Retrieval
Manufacturing documents contain both natural language and structured identifiers such as part numbers and standard codes. A vector-only retriever is not sufficient. BM25 is used alongside embeddings so exact matches are not missed.

### Jina Embeddings
The selected embedding model supports German and English. That is important for manufacturing material, manuals, and standards that may appear in both languages.

### ChromaDB
ChromaDB provides a local persistent vector store with low setup cost. That makes it a good fit for iteration and demo deployment.

### Answer Modes
The production default is local extractive answering. This keeps the public demo deterministic and avoids external API quota or cloud model-hosting requirements.

Gemini remains available as an optional cloud LLM when a key is configured. Ollama remains available for local development and comparison runs on machines with enough resources.

## Current Constraints

- The current workspace uses a public 4-PDF manufacturing corpus.
- Chroma and BM25 indexes are present locally, but Render deployment still needs a live service verification.
- The current evaluation runner collects answers and sources; it does not currently compute RAGAS metrics.
- OCR support will be added only if text extraction from real PDFs shows a gap.
