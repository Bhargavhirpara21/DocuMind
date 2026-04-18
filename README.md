# DocuMind

DocuMind is a retrieval-augmented generation system for manufacturing documents. It answers technical questions from PDFs such as datasheets, machine manuals, standards previews, and tool catalogs with cited source passages.

## Current Status

- Phase 1: Core ingestion modules implemented
- Phase 2: Query pipeline implemented
- Phase 3: FastAPI and Streamlit scaffolding implemented
- GitHub repository published on `main`
- PDFs still need to be added to `data/pdfs/` before end-to-end ingestion can run

## Features

- PDF loading and chunking with page metadata
- Local embedding generation using Jina embeddings
- Persistent vector storage with ChromaDB
- BM25 keyword retrieval for exact part numbers and standards codes
- Hybrid retrieval with reciprocal-rank fusion
- LLM abstraction for Gemini and Ollama
- FastAPI endpoints for question answering, ingestion, upload, and document listing
- Streamlit UI for chat-style querying and PDF upload
- Unit tests for config, prompt, loader, and BM25 persistence

## Architecture

1. PDFs are loaded from `data/pdfs/`.
2. Documents are chunked with metadata preserved.
3. Chunks are embedded and stored in ChromaDB.
4. A BM25 index is saved for exact keyword retrieval.
5. Queries use hybrid retrieval to combine semantic and keyword search.
6. Retrieved chunks are sent to an LLM with a grounded prompt.
7. The answer is returned with source metadata.

## Project Structure

```text
DocuMind/
├── src/
│   ├── api/
│   ├── embeddings/
│   ├── generation/
│   ├── ingestion/
│   ├── pipeline/
│   └── retrieval/
├── frontend/
├── tests/
├── data/
├── requirements.txt
└── README.md
```

## Local Setup

1. Create and activate a Python 3.12 virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the tests:

```bash
python -m pytest tests -v
```

4. Add PDFs to `data/pdfs/`.
5. Run ingestion:

```bash
python -m src.pipeline.ingest
```

6. Start the API:

```bash
uvicorn src.api.routes:app --reload --port 8000
```

7. Start the UI:

```bash
streamlit run frontend/app.py
```

## API Endpoints

- `GET /api/health`
- `POST /api/ask`
- `POST /api/ingest`
- `GET /api/documents`
- `POST /api/upload`

## Testing

The current test suite covers:

- configuration values and directory creation
- prompt formatting
- PDF chunk metadata behavior
- BM25 save/load round-trip

Run:

```bash
python -m pytest tests -v
```

## Next Milestones

- Add PDF source files to `data/pdfs/`
- Validate ingestion end to end
- Expand tests for retrieval and query behavior
- Add evaluation dataset and RAGAS scoring
- Add Docker and deployment hardening

## License

No license has been selected yet.