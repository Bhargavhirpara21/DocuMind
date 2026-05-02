# Plan: DocuMind — RAG-Powered Manufacturing Document Q&A System

**TL;DR**: Build a retrieval-augmented generation (RAG) system that answers natural language questions from manufacturing documents (ISO standards, material datasheets, machine manuals, cutting tool catalogs). Users type a question, the system finds relevant passages using hybrid search, sends them to an LLM, and returns an answer with exact citations. Python backend, Streamlit frontend, deployed on Render.

**Why this project is the right fit:**
- Demonstrates RAG expertise (the most in-demand AI engineering skill in 2026)
- Domain-specific for manufacturing/engineering (matches your Walter Tools experience)
- Hybrid retrieval shows you understand real-world search challenges (part numbers, ISO codes)
- Bilingual German-English support (relevant for German companies)
- Production-grade: API + UI + Docker + evaluation + cloud deployment
- Highly discussable in interviews: chunking strategy, embedding choice, retrieval trade-offs

---

## Architecture

```
Phase 1: Ingestion (one-time)

  PDF files  -->  LlamaIndex Loader  -->  SentenceSplitter  -->  Jina Embeddings  -->  ChromaDB
  (data/pdfs)    (extract text)          (512 tokens,           (text to 768d        (vector store)
                                          50 overlap)            vectors)
                                              |
                                              v
                                         BM25 Index
                                        (keyword search)

Phase 2: Query (every question)

  User Question  -->  Hybrid Retriever  -->  Top 5 Chunks  -->  Direct Rules / Answer Mode  -->  Answer + Citations
                      (vector + BM25)       with metadata       (local default; Gemini or     (source doc + page)
                                                                 Ollama optional)

Infrastructure:

  FastAPI (REST API)  |  Streamlit (Demo UI)  |  Docker  |  Render  |  Answer Collection (Evaluation)
```
---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Language | Python 3.12 | Your primary language |
| RAG Framework | LlamaIndex | Purpose-built for RAG, less code than LangChain |
| Embedding Model | jina-embeddings-v2-base-de | Bilingual German-English, 768 dims, 8192 token context |
| Vector Database | ChromaDB | Simple setup, good for prototyping |
| Keyword Search | BM25 (via rank_bm25) | Exact matching for part numbers, ISO codes |
| Production answer mode | Local extractive answering | No external quota, deterministic public demo behavior |
| Optional cloud LLM | Google Gemini 2.5 Flash | Useful for comparison when an API key is available |
| Local LLM option | Ollama + Gemma | Local experimentation on machines with enough resources |
| Backend API | FastAPI + Uvicorn | You already know it, fast, auto-docs |
| Frontend | Streamlit | Quick demo UI, free cloud hosting option |
| Evaluation | Answer collection runner | Repeatable question set with expected answers and sources |
| Containerization | Docker + Docker Compose | Reproducible deployment |
| Cloud | Render | Real cloud deployment experience |
| Version Control | Git + GitHub | Code hosting, CI/CD |

---

## Documents (Data Sources)

Publicly available manufacturing PDFs. No Walter Tools internal data.

| Type | Source | Count |
|------|--------|-------|
| Material datasheets | ThyssenKrupp, Dillinger, Sandvik public sites | 2-3 |
| Cutting tool catalogs | Walter Tools / Sandvik Coromant public catalogs | 2-3 |
| CNC machine manuals | Haas Automation (publicly available) | 1-2 |
| ISO/DIN standard previews | iso.org, Beuth Verlag free previews | 2-3 |

**Total: 10+ PDFs**

---

## Phase 1: Ingestion Pipeline (Days 1-3)

### File: src/config.py
**Purpose**: Central settings file. Reads secrets from .env, stores all configuration in one place.

**What it contains**:
- LLM_PROVIDER, GEMINI_API_KEY, OLLAMA_MODEL (from .env)
- PDF_DIR = data/pdfs
- CHROMA_DIR = data/chroma
- CHUNK_SIZE = 512
- CHUNK_OVERLAP = 50
- EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-de"
- TOP_K = 5 (number of chunks to retrieve per query)

**Status**: DONE

---

### File: src/ingestion/loader.py
**Purpose**: Read PDFs and split into chunks. Entry point for all documents.

**Functions**:
- `load_pdfs(pdf_dir: Path) -> list[Document]` -- reads all PDFs from a directory, returns list of Document objects with text and metadata (filename, page number)
- `chunk_documents(documents, chunk_size, chunk_overlap) -> list[TextNode]` -- splits documents into chunks using SentenceSplitter, preserves metadata

**Packages**: llama-index-core, llama-index-readers-file

**Status**: DONE

---

### File: src/embeddings/embedder.py
**Purpose**: Convert text chunks into number-vectors (embeddings) so ChromaDB can search by meaning.

**Functions**:
- `get_embedding_model() -> HuggingFaceEmbedding` -- loads and returns the Jina embedding model
- `embed_texts(texts: list[str]) -> list[list[float]]` -- converts a list of text strings into vectors (each vector is 768 numbers)

**Packages**: llama-index-embeddings-huggingface, sentence-transformers

**Key detail**: The Jina model runs locally on your laptop (no API needed). It downloads once (~500MB) and is cached. Each text chunk becomes a list of 768 numbers that represent its meaning.

**Test**: Run the file directly. It should embed a sample text and print the vector length (768).

---

### File: src/retrieval/vector_store.py
**Purpose**: Set up ChromaDB to store and search vectors.

**Functions**:
- `get_vector_store() -> ChromaVectorStore` -- creates or connects to ChromaDB collection at CHROMA_DIR
- `build_index(chunks, embed_model) -> VectorStoreIndex` -- takes chunks + embedding model, embeds all chunks, stores in ChromaDB
- `load_index(embed_model) -> VectorStoreIndex` -- loads existing index from ChromaDB (for querying without re-ingesting)

**Packages**: llama-index-vector-stores-chroma, chromadb

**Key detail**: ChromaDB saves data to disk at data/chroma/. Once you ingest documents, the vectors persist. You don't need to re-ingest every time you restart the app.

**Test**: Run after ingestion. Print number of vectors stored in ChromaDB.

---

### File: src/retrieval/bm25.py
**Purpose**: Build a keyword search index for exact matching (part numbers, ISO codes, material grades).

**Functions**:
- `build_bm25_index(chunks) -> BM25Retriever` -- creates a BM25 index from the chunk texts
- `save_bm25_index(chunks, path)` -- saves the BM25 index to disk for later use
- `load_bm25_index(path) -> BM25Retriever` -- loads saved BM25 index

**Packages**: rank_bm25 (or llama-index built-in BM25)

**Key detail**: BM25 searches by exact word matching. When a user asks about "CNMG120408", BM25 finds chunks containing that exact string. Vector search would fail here because embeddings treat part numbers as meaningless character sequences.

**Test**: Build index, search for a known part number from your PDF, verify it returns the right chunk.

---

### File: src/pipeline/ingest.py
**Purpose**: Ties the entire ingestion pipeline together. One script that does everything: load PDFs, chunk, embed, store in ChromaDB, build BM25 index.

**Flow**:
```
1. Load all PDFs from data/pdfs/ (using loader.py)
2. Split into chunks (using loader.py)
3. Load embedding model (using embedder.py)
4. Store chunks + vectors in ChromaDB (using vector_store.py)
5. Build and save BM25 index (using bm25.py)
6. Print summary: X PDFs loaded, Y chunks created, Z vectors stored
```

**Usage**: `python -m src.pipeline.ingest`

**Key detail**: You run this once. After it finishes, all your documents are searchable. You don't run it again unless you add new PDFs.

**Test**: Run it. Check that data/chroma/ folder is created with files inside.

---

## Phase 2: Query Pipeline (Days 4-6)

### File: src/retrieval/hybrid.py
**Purpose**: Combine vector search (ChromaDB) and keyword search (BM25) results into one ranked list.

**Functions**:
- `get_hybrid_retriever(vector_index, bm25_retriever) -> QueryFusionRetriever` -- creates a retriever that runs both vector and BM25 search, then merges results using reciprocal rank fusion
- `retrieve(query: str) -> list[NodeWithScore]` -- takes a question, returns top K relevant chunks with scores

**Packages**: llama-index-core (QueryFusionRetriever)

**How hybrid works**:
```
User question: "What cutting speed for CNMG120408 in steel?"

Vector search returns:     BM25 search returns:
1. Cutting speed guide     1. CNMG120408 spec sheet  <-- exact match
2. Steel machining tips    2. CNMG product family
3. General insert guide    3. Insert specifications

Hybrid merges both lists using reciprocal rank fusion:
Final result:
1. CNMG120408 spec sheet  (high in BM25 + decent in vector)
2. Cutting speed guide     (high in vector)
3. Steel machining tips    (medium in both)
4. CNMG product family     (medium in BM25)
5. General insert guide    (medium in vector)
```

**Test**: Search for a part number -- verify BM25 finds it. Search for a concept -- verify vector search finds it.

---

### File: src/generation/llm.py
**Purpose**: Connect to the LLM (Gemini API or Ollama local). Abstracts the LLM choice so the rest of the code doesn't care which LLM is being used.

**Functions**:
- `get_llm() -> LLM` -- reads LLM_PROVIDER from config, returns the appropriate LLM instance

**Logic**:
```python
if config.LLM_PROVIDER == "gemini":
    return Gemini(model="models/gemini-2.5-flash", api_key=config.GEMINI_API_KEY)
elif config.LLM_PROVIDER == "ollama":
    return Ollama(model=config.OLLAMA_MODEL)
```

**Packages**: llama-index-llms-gemini, llama-index-llms-ollama (install ollama package only when needed)

**Test**: Run the file directly. Send a simple question ("What is 2+2?") and print the response.

---

### File: src/generation/prompt.py
**Purpose**: Define the prompt template that instructs the LLM how to answer questions using the retrieved chunks.

**Contains**:
- `QA_PROMPT` -- the prompt template with placeholders for context (retrieved chunks) and question

**Prompt template** (this is critical for answer quality):
```
You are a manufacturing engineering assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I cannot find this information in the available documents."

Rules:
- Be precise and technical
- Include specific numbers, values, and specifications when available
- Always cite the source document and page number
- If the context is in German, you may answer in German or English based on the question language

Context:
{context}

Question: {question}

Answer:
```

**Key detail**: The prompt tells the LLM to only use information from the retrieved chunks, never make things up. This is what prevents hallucination. The citation instruction ensures every answer includes its source.

---

### File: src/pipeline/query.py
**Purpose**: Ties the entire query pipeline together. Takes a user question and returns an answer with citations.

**Functions**:
- `setup_query_engine() -> QueryEngine` -- loads the index, BM25, hybrid retriever, LLM, and prompt. Returns a ready-to-use query engine
- `ask(question: str) -> dict` -- takes a question, returns {"answer": "...", "sources": [{"document": "...", "page": N, "text": "..."}]}

**Flow**:
```
1. User asks: "What is the tensile strength of Ti-6Al-4V?"
2. Hybrid retriever searches ChromaDB + BM25
3. Top 5 chunks returned with metadata
4. Chunks + question formatted using prompt template
5. Direct-answer rules try known deterministic extraction cases
6. If no rule matches, the selected answer provider generates an answer
7. Citations extracted from chunk metadata
8. Return: answer + list of sources (document name, page number, relevant text)
```

**Usage**: `python -m src.pipeline.query "What is the tensile strength of Ti-6Al-4V?"`

**Test**: Ask 5 different questions. Verify answers are correct, citations point to the right documents.

---

## Phase 3: API + Frontend (Days 7-9)

### File: src/api/routes.py
**Purpose**: FastAPI REST API exposing the query pipeline as HTTP endpoints.

**Endpoints**:

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | /api/health | Health check | - | `{"status": "ok"}` |
| POST | /api/ask | Ask a question | `{"question": "..."}` | `{"answer": "...", "sources": [...]}` |
| POST | /api/ingest | Trigger ingestion | `{"pdf_dir": "..."}` (optional) | `{"status": "ingested", "chunks": N}` |
| GET | /api/documents | List ingested documents | - | `[{"name": "...", "pages": N}]` |
| POST | /api/upload | Upload a PDF and ingest it | multipart file | `{"status": "uploaded", "chunks": N}` |

**Packages**: fastapi, uvicorn, python-multipart

**Run**: `uvicorn src.api.routes:app --reload --port 8000`

**Auto-docs**: FastAPI generates API documentation at http://localhost:8000/docs

---

### File: frontend/app.py
**Purpose**: Streamlit demo UI with two pages.

**Page 1: "Ask Questions" (default)**
- Chat-style interface
- Text input for questions
- Displays answer with formatted citations below
- Shows which documents are loaded in a sidebar
- Conversation history (previous Q&A pairs stay visible)

**Page 2: "Upload & Ask"**
- File upload widget (drag & drop PDF)
- After upload: progress bar showing ingestion
- Then same chat interface as Page 1, but querying the uploaded document

**Layout**:
```
┌─────────────────────────────────────────────────┐
│  DocuMind - Manufacturing Document Q&A          │
│                                                  │
│  Sidebar:                    Main Area:          │
│  ┌──────────┐               ┌─────────────────┐ │
│  │ Page:    │               │ Ask a question:  │ │
│  │ • Ask    │               │ [____________]   │ │
│  │ • Upload │               │                  │ │
│  │          │               │ Answer:          │ │
│  │ Loaded:  │               │ The tensile...   │ │
│  │ • doc1   │               │                  │ │
│  │ • doc2   │               │ Sources:         │ │
│  │ • doc3   │               │ 📄 datasheet.pdf │ │
│  │          │               │    page 12       │ │
│  └──────────┘               └─────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Packages**: streamlit

**Run**: `streamlit run frontend/app.py`

---

## Phase 4: Evaluation (Days 10-11)

### File: evaluation/test_questions.json
**Purpose**: 30-50 test questions with expected answers for measuring system quality.

**Format**:
```json
[
  {
    "question": "What is the tensile strength of Ti-6Al-4V?",
    "expected_answer": "950 MPa",
    "source_document": "material_datasheet.pdf",
    "source_page": 12,
    "type": "factual_lookup"
  },
  {
    "question": "What does DIN 7991 specify?",
    "expected_answer": "Countersunk head screws with hexalobular drive",
    "source_document": "din_standards.pdf",
    "source_page": 1,
    "type": "explanation"
  }
]
```

**Key detail**: You create these manually by reading your PDFs and writing questions with known answers. This is the ground truth used to review collected answers and sources.

---

### File: evaluation/evaluate.py
**Purpose**: Run the evaluation question set through the current query pipeline and collect answers, sources, and expected-answer metadata.

**What it does**:
```
1. Load test questions from test_questions.json
2. Run each question through the query pipeline
3. Save each response with retrieved sources, expected answer, expected source document, expected source page, and question type.
4. Compare the collected output manually or with a future scoring script.
```

RAGAS can still be added later if metric scoring becomes a project requirement.

---

## Phase 5: Docker + Deployment (Days 12-14)

### Deployment Runtime Decision

Use Render web services as the cloud deployment path.

Why this path:
- It matches the existing `Dockerfile` and local `docker-compose.yml`.
- It packages the API and Streamlit UI into a repeatable artifact.
- It is easier to document, restart, and move between environments than a manual VM setup.

Recommended layout:
- API web service with a persistent disk mounted at `/app/data`.
- Frontend web service pointed at the API service URL.

Fallback only if Render is blocked by policy or ops constraints:
- Plain Python service on a VM with the same app code and environment variables.

### Deployment LLM Decision

Use the local extractive answer mode in production (`LLM_PROVIDER=local`).

Why this path:
- It does not depend on Gemini quota or a separate paid API key.
- It does not require running Ollama or downloading a model on the cloud host.
- It is deterministic and stable for the public demo link.

Optional developer-only alternatives:
- Gemini for ad hoc testing when a key is available.
- Ollama for local experimentation on a machine with enough resources.

### File: Dockerfile
**Purpose**: Packages the entire application into a container.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
```

---

### File: docker-compose.yml
**Purpose**: Runs all services together with one command.

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
      - "8501:8501"
    env_file: .env
    volumes:
      - ./data:/app/data
```

**Usage**: `docker-compose up --build`

---

### Render Deployment (Web Services)

**Steps**:
1. Create a Render account and connect your GitHub repo.
2. Create an API web service from this repository.
3. Set the API start command to `uvicorn src.api.routes:app --host 0.0.0.0 --port $PORT`.
4. Attach a persistent disk to the API service and mount it at `/app/data`.
5. Copy `.env.example` into the service environment and keep `LLM_PROVIDER=local`.
6. Create a Streamlit web service from the same repository.
7. Set the Streamlit start command to `streamlit run frontend/app.py --server.address 0.0.0.0 --server.port $PORT`.
8. Point `DOCUMIND_API_URL` in the frontend service at the Render API service URL.

**Result**: A public Render URL for the frontend, with durable API storage for Chroma, BM25, and uploads.

---

## Phase 6: Polish (Days 15-16)

### File: README.md
**Purpose**: Professional documentation. Must include:

1. Project title + one-line description
2. Live demo link (Render URL)
3. Architecture diagram (ASCII or image)
4. Features list
5. Tech stack table
6. Quick start (local + Docker)
7. API endpoints table
8. Evaluation results from the collected-answer runner
9. Screenshots or demo GIF
10. Project structure
11. License

**Reference**: Use your DevHealth README as a template for formatting and structure.

---

### File: tests/
**Purpose**: Unit tests for critical components.

| Test File | What it tests |
|-----------|--------------|
| test_loader.py | PDF loading returns documents, chunking produces correct number of chunks, metadata is preserved |
| test_config.py | Settings load correctly, paths exist |
| test_retrieval.py | Vector search returns results, BM25 returns results, hybrid combines both |
| test_pipeline.py | End-to-end: question in, answer + citations out |

**Packages**: pytest

**Run**: `pytest tests/ -v`

---

## Project Structure

```
documind/
├── README.md                      # Professional documentation
├── .env.example                   # Template for secrets
├── .env                           # Actual secrets (not in Git)
├── .gitignore                     # Python template
├── requirements.txt               # All Python dependencies
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Multi-service orchestration
│
├── docs/
│   ├── DESIGN.md                  # Technical design decisions
│   └── ARCHITECTURE.md            # Architecture notes
│
├── data/
│   ├── pdfs/                      # Manufacturing PDFs (not in Git)
│   └── chroma/                    # ChromaDB storage (not in Git)
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # ✅ DONE - Central settings
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py              # ✅ DONE - PDF loading + chunking
│   │   └── chunker.py             # (merged into loader.py)
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py            # Jina embedding model
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py        # ChromaDB setup and operations
│   │   ├── bm25.py                # BM25 keyword search
│   │   └── hybrid.py              # Hybrid retriever (vector + BM25)
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm.py                 # LLM connection (Gemini / Ollama)
│   │   └── prompt.py              # Prompt templates
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── ingest.py              # Full ingestion pipeline
│   │   └── query.py               # Full query pipeline
│   │
│   └── api/
│       ├── __init__.py
│       └── routes.py              # FastAPI endpoints
│
├── frontend/
│   └── app.py                     # Streamlit demo UI
│
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_loader.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
│
├── evaluation/
│   ├── test_questions.json        # Ground truth Q&A pairs
│   └── evaluate.py                # Evaluation answer-collection script
│
└── scripts/
    └── ingest.py                  # CLI entry point for ingestion
```

---

## Build Order

Build and test each file in this exact order:

| Order | File | What to verify |
|-------|------|---------------|
| 1 | src/config.py | ✅ DONE - prints all settings correctly |
| 2 | src/ingestion/loader.py | ✅ DONE - loads 452 pages, creates 672 chunks |
| 3 | src/embeddings/embedder.py | Embeds sample text, prints vector of length 768 |
| 4 | src/retrieval/vector_store.py | Stores chunks in ChromaDB, data/chroma/ folder created |
| 5 | src/retrieval/bm25.py | Searches by keyword, finds correct chunks |
| 6 | src/pipeline/ingest.py | Runs full ingestion: load, chunk, embed, store |
| 7 | src/generation/llm.py | Selects local, Gemini, or Ollama answer provider |
| 8 | src/generation/prompt.py | Prompt template renders correctly with test data |
| 9 | src/retrieval/hybrid.py | Hybrid search returns results from both vector and BM25 |
| 10 | src/pipeline/query.py | End-to-end: question -> answer with citations |
| 11 | src/api/routes.py | All API endpoints work via http://localhost:8000/docs |
| 12 | frontend/app.py | Streamlit UI shows, can ask questions, sees citations |
| 13 | tests/ | All pytest tests pass |
| 14 | evaluation/ | Evaluation answers collected for all test questions |
| 15 | Dockerfile + docker-compose.yml | docker-compose up works, app accessible |
| 16 | Render deployment | App accessible via public Render URL |
| 17 | README.md | Complete documentation with demo link, screenshots |

---

## Decisions Log

| Decision | Chosen | Why | Alternative |
|----------|--------|-----|-------------|
| RAG framework | LlamaIndex | Purpose-built for RAG, 30-40% less code | LangChain |
| Embedding model | jina-embeddings-v2-base-de | Bilingual DE/EN, 768d, 8192 token context | OpenAI embeddings (paid) |
| Vector database | ChromaDB | Simple setup, good for prototyping | Qdrant, pgvector |
| Production answer mode | Local extractive answering | Stable no-quota cloud behavior | Gemini, Ollama |
| Optional cloud LLM | Gemini 2.5 Flash | Good comparison mode when a key is available | GPT-4o (paid), Ollama (local) |
| Local LLM option | Ollama + Gemma | Local experimentation | None |
| Retrieval strategy | Hybrid (vector + BM25) | Manufacturing docs need exact code matching | Vector only |
| Backend | FastAPI | Fast, auto-docs, you know it well | Flask |
| Frontend | Streamlit | Quick demo UI, free cloud hosting | React (overkill for demo) |
| Evaluation | Answer collection runner | Repeatable regression set without LLM-judge dependency | RAGAS later |
| Cloud | Render | Real cloud deployment experience for CV | Self-managed VM |
| Chunking | 512 tokens, 50 overlap | Balance of precision and context | 256 or 1024 |

---

## Verification Checklist

Before considering the project complete:

1. **Ingestion works**: 10+ PDFs loaded, chunks stored in ChromaDB
2. **Vector search works**: Semantic question returns relevant chunks
3. **BM25 search works**: Part number query returns exact matches
4. **Hybrid search works**: Combines both for better results than either alone
5. **LLM generates answers**: Correct, grounded in retrieved chunks
6. **Citations work**: Every answer shows source document + page number
7. **Upload works**: User can upload a new PDF and query it
8. **API works**: All endpoints respond correctly (test via /docs)
9. **UI works**: Streamlit shows questions, answers, and citations cleanly
10. **Docker works**: docker-compose up starts everything
11. **Render works**: App accessible via public URL
12. **Evaluation output**: Collected answers and sources reviewed against expected answers
13. **Tests pass**: pytest tests/ passes with no failures
14. **README is complete**: Someone can understand, run, and deploy from README alone
15. **Demo GIF**: Shows the system in action (record with ScreenToGif or similar)
