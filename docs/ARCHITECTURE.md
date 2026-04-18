# Architecture

## Overview

DocuMind is a RAG system for manufacturing PDFs. It is organized around two main flows: ingestion and query.

## Ingestion Flow

- Load PDF files from `data/pdfs/`
- Extract text and preserve document metadata
- Split text into chunks
- Generate embeddings for each chunk
- Store chunks in ChromaDB
- Build and save a BM25 index for exact-match retrieval

## Query Flow

- Receive a question from the API or UI
- Retrieve top chunks using hybrid search
- Format a grounded prompt with context and question
- Send the prompt to the selected LLM
- Return an answer with source metadata

## Design Choices

- Use LlamaIndex for ingestion and vector-store integration
- Use Jina embeddings for bilingual German-English support
- Use ChromaDB for local persistent vector storage
- Use BM25 for code-heavy manufacturing terms
- Use hybrid retrieval because technical documents often require both semantic search and exact matching
- Keep the LLM provider abstract so Gemini and Ollama can be swapped without changing the pipeline

## Operational Notes

- OCR is not enabled by default.
- It should be introduced only if scanned PDFs are discovered.
- The ingestion pipeline should be rerun only when documents change.
- The query pipeline should stay read-only against the stored indexes.
