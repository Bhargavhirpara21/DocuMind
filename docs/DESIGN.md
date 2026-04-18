# Design Decisions

## Goals

- Answer questions from manufacturing documents with citations
- Support both exact-code lookups and semantic questions
- Keep the system simple enough to run locally and deploy to AWS

## Key Decisions

### LlamaIndex
LlamaIndex is used because the project is fundamentally a RAG system. It reduces boilerplate around document loading, chunking, retrievers, and vector store integration.

### Hybrid Retrieval
Manufacturing documents contain both natural language and structured identifiers such as part numbers and standard codes. A vector-only retriever is not sufficient. BM25 is used alongside embeddings so exact matches are not missed.

### Jina Embeddings
The selected embedding model supports German and English. That is important for manufacturing material, manuals, and standards that may appear in both languages.

### ChromaDB
ChromaDB provides a local persistent vector store with low setup cost. That makes it a good fit for iteration and demo deployment.

### Gemini and Ollama
The system should work with a cloud LLM or a local fallback. That keeps the project usable even if API access is limited.

## Current Constraints

- No PDFs have been loaded yet.
- End-to-end ingestion and query validation are blocked until source PDFs are added.
- OCR support will be added only if text extraction from real PDFs shows a gap.