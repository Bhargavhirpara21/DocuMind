# DocuMind

DocuMind is a retrieval-augmented generation system for manufacturing documents. It answers technical questions from PDFs such as datasheets, machine manuals, standards previews, and tool catalogs with cited source passages.

## Current Status

- Core ingestion, hybrid retrieval, FastAPI, and Streamlit are implemented
- The current workspace uses a 4-PDF public manufacturing corpus
- ChromaDB and BM25 indexes are rebuilt locally and the live ask flow works
- GitHub repository is published on `main`
- Remaining work is deployment hardening, final docs polish, and optional corpus expansion

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
├── evaluation/
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

4. Put your PDFs in `data/pdfs/` if you want to change the corpus. The repository already includes a working sample set.
5. Run ingestion:

```bash
python -m src.pipeline.ingest --reset-indexes
```

Use `--max-documents N` if you want a smaller smoke run while iterating.

6. Start the API:

```bash
uvicorn src.api.routes:app --reload --port 8000
```

7. Start the UI:

```bash
streamlit run frontend/app.py
```

## Docker

Build and run the API plus frontend with Docker Compose:

```bash
docker compose up --build
```

- API: http://localhost:8000
- Streamlit UI: http://localhost:8501

The frontend container talks to the API through the `api` service name.

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

## Evaluation

The starter evaluation set lives in `evaluation/test_questions.json`.

Run:

```bash
python evaluation/evaluate.py
```

The script writes `evaluation/results.json` locally.

## Next Milestones

- Expand or swap the PDF corpus if you want broader coverage
- Finalize the evaluation set and record repeatable quality metrics
- Validate Docker Compose end to end
- Add AWS deployment hardening and a public demo URL
- Polish the README with screenshots, a live demo link, and final evaluation results

## License

No license has been selected yet.