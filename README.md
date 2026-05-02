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
- Local no-quota answer mode, plus optional Gemini and Ollama
- FastAPI endpoints for question answering, ingestion, upload, and document listing
- Streamlit UI for chat-style querying and PDF upload
- Unit tests for config, prompt, loader, and BM25 persistence

## Deployment Default

The cloud deployment uses local extractive mode (`LLM_PROVIDER=local`) as the default production answer path.
That keeps the public demo stable and avoids external quota or model-hosting dependencies.

Gemini and Ollama stay available for local experimentation and future comparison, but they are not required for the cloud path.

## Architecture

1. PDFs are loaded from `data/pdfs/`.
2. Documents are chunked with metadata preserved.
3. Chunks are embedded and stored in ChromaDB.
4. A BM25 index is saved for exact keyword retrieval.
5. Queries use hybrid retrieval to combine semantic and keyword search.
6. The query path first tries deterministic direct-answer rules for known brochure questions.
7. If no direct rule applies, retrieved chunks are sent to the configured answer provider with a grounded prompt.
8. The answer is returned with source metadata.

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

6. To try local Gemma through Ollama, create a `.env` file with:

```ini
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma3:4b
```

Then pull the model once:

```bash
ollama pull gemma3:4b
```

`gemma3:4b` is the current quality-first choice in this workspace; if you want more speed, try `gemma3:1b`, and if you want a smaller balanced model, you can switch to `gemma2:2b`.

For the Windows-friendly copy-paste run guide I have been using in this workspace, see [docs/GETTING_STARTED_WINDOWS.md](docs/GETTING_STARTED_WINDOWS.md).

7. Start the API:

```bash
uvicorn src.api.routes:app --reload --port 8000
```

If Gemma is not satisfying, delete or change `.env` to switch back to `LLM_PROVIDER=local` or any other provider.

8. Start the UI:

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

For a cloud host, copy [.env.example](.env.example) to `.env` on the server and keep `LLM_PROVIDER=local` so the public demo does not depend on a separate model API or a local Ollama instance.

For the full cloud deployment checklist, see [docs/DEPLOYMENT_RENDER.md](docs/DEPLOYMENT_RENDER.md).

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
- hybrid retrieval ranking rules for page-aware questions
- deterministic direct-answer behavior in the query path
- API health, document listing, and upload ingest wiring

Run:

```bash
python -m pytest tests -v
```

## Evaluation

The evaluation question set lives in `evaluation/test_questions.json`.

Run:

```bash
python evaluation/evaluate.py
```

The script writes collected answers and sources to `evaluation/results.json` locally. It is a repeatable answer-collection runner, not a current RAGAS metric scorer.

## Next Milestones

- Expand or swap the PDF corpus if you want broader coverage
- Finalize the evaluation set and record repeatable quality metrics
- Validate Docker Compose end to end
- Add Render deployment hardening and a public demo URL
- Polish the README with screenshots, a live demo link, and final evaluation results

## License

No license has been selected yet.
