# Jogi Ayu RAG — Ayurvedic Medicine Educational Assistant

A Retrieval-Augmented Generation (RAG) system for Ayurvedic medicine education. Ingests YouTube transcripts, educational books, and reference documents via a web GUI or CLI, then answers questions with citation-grounded responses.

## Architecture

```
                    QUERY
                      │
      ┌───────────────┼───────────────┐
      ▼                               ▼
 Sparse Retrieval                Dense Retrieval
 (FTS5 BM25, disk-backed)        (bge-small + Qdrant)
      │                               │
      └───────────┬───────────────────┘
                  ▼
           RRF Fusion
         (Reciprocal Rank Fusion)
                  │
                  ▼
          Cross-Encoder Rerank
            (ms-marco-MiniLM)
                  │
                  ▼
          Document Assembly
          (DB parent lookups)
                  │
                  ▼
          LLM Generation
    (OpenAI-compatible API)
                  │
                  ▼
         Cited Answer Output
```

### Pipeline Stages

| # | Stage | Module | Description |
|---|-------|--------|-------------|
| 1 | Sparse Retrieval | `sparse_fts.py` | BM25 keyword search via SQLite FTS5, disk-backed with near-zero RAM |
| 2 | Dense Retrieval | `indexing.py` | Semantic search via `bge-small-en-v1.5` embeddings stored in Qdrant |
| 3 | RRF Fusion | `retrieval.py` | Reciprocal Rank Fusion combining sparse + dense rankings |
| 4 | Reranking | `reranking.py` | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores fusion candidates |
| 5 | Document Assembly | `reranking.py` | Groups chunks under their parent documents via DB lookups |
| 6 | Generation | `generation.py` | OpenAI-compatible LLM call with Ayurveda-aware, citation-grounded answers |

### Data Model

Documents are chunked into a 2-level hierarchy:

- **Document** — a source file (transcript, book chapter, PDF, pasted text)
- **Chunk** — a text fragment produced by the hybrid chunking strategy

Chunks are stored in SQLite (`data/chunks.db`) with prev/next/parent ID chains for navigation and enriched with Ayurvedic metadata (doshas, symptoms, treatments, keywords).

### Hybrid Chunking Strategy

New in v2. When ingesting unstructured text (transcripts, books without rigid structure), the chunker combines:

1. **Paragraph structure awareness** — detects paragraph boundaries and heading-like lines
2. **Semantic splitting** — uses embedding cosine similarity to find natural topic boundaries between paragraphs
3. **Token overlap** — configurable overlap window to maintain context across chunk boundaries
4. **LLM-added metadata** — optional rule-based extraction of dosha references, symptom keywords, and treatment terms per chunk

## Files

| File | Role |
|------|------|
| `main.py` | CLI query entry: loads indices, runs query, prints cited answer |
| `cli_ingestion.py` | CLI ingestion: `ingest`, `status`, `list`, `delete`, `rebuild` commands |
| `index_data.py` | Batch index builder from a directory of source files |
| `chunking.py` | `HybridChunker` — paragraph detection, semantic splitting, overlap, LLM metadata |
| `db.py` | SQLite chunk store + documents table with CRUD and lookup methods |
| `ingestion/file_readers.py` | Parsers for `.txt`, `.md`, `.pdf`, `.json`, `.csv` files |
| `ingestion/connectors.py` | `FileConnector`, `DirectoryConnector`, `RawTextConnector`, `WebUploadConnector` |
| `ingestion/pipeline.py` | Orchestration: read → chunk → sparse index → dense index |
| `indexing.py` | `SparseRetriever` (legacy BM25), `DenseRetriever` (Qdrant + fastembed) |
| `sparse_fts.py` | `SparseFTS5Retriever` — modern disk-backed BM25 via SQLite FTS5 |
| `retrieval.py` | Hybrid retrieval with RRF fusion and rerank orchestration |
| `reranking.py` | Cross-encoder reranker and document context assembly |
| `generation.py` | OpenAI-compatible LLM client with citation validation and Ayurveda-aware prompts |
| `multi_hop.py` | Multi-hop orchestrator for complex questions across retrieval rounds |
| `api.py` | FastAPI server: query API, ingestion API, document management, serves React UI |
| `ui/` | React (Vite) frontend: chat with citations, data explorer, ingestion panel |
| `benchmark.py` | Layer-by-layer benchmarking and quality evaluation |
| `docker-compose.yml` | Docker Compose config for Qdrant container |
| `requirements.txt` | Python dependencies |

## Setup

### Prerequisites

- Python 3.10+
- (Optional) OpenAI-compatible API endpoint for generation
- (Optional) Docker + Docker Compose for remote Qdrant mode
- (Optional) Node.js 18+ for the web UI

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### Environment

Copy `.env.example` to `.env` and configure:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

- `QDRANT_URL` — set to connect to Docker Qdrant; leave unset for local disk mode
- `QDRANT_API_KEY` — only needed for Qdrant Cloud or gated instances
- `OPENAI_BASE_URL` — can point to any OpenAI-compatible endpoint (vLLM, Ollama, LiteLLM, etc.)

### Docker Qdrant (remote mode)

```bash
docker compose up -d
```

Qdrant will be available at `http://localhost:6333` (HTTP) and `localhost:6334` (gRPC).

## Usage

### 1. Ingest Data

**Via CLI:**

```bash
# Ingest a single file
python cli_ingestion.py ingest --file data/raw/transcript.txt

# Ingest a directory of files
python cli_ingestion.py ingest --dir data/raw/

# Ingest raw text directly
python cli_ingestion.py ingest --text "Vata dosha governs movement..." --title "Vata Overview"

# Ingest with LLM metadata generation
python cli_ingestion.py ingest --file lecture.pdf --llm-metadata

# Tune chunking parameters
python cli_ingestion.py ingest --dir data/raw/ --chunk-size 256 --chunk-overlap 48
```

**Via Web GUI:**

```bash
python api.py
# Open http://localhost:8000 → Ingest tab → Upload files or paste text
```

**Supported file formats:** `.txt`, `.md`, `.pdf`, `.json`, `.csv`, `.log`

**Batch indexing from a directory:**

```bash
python index_data.py --data-dir data/raw/
```

### 2. Manage Data

```bash
# View ingestion status
python cli_ingestion.py status

# List all documents
python cli_ingestion.py list

# Delete a document and its chunks
python cli_ingestion.py delete <doc_id>

# Delete all indices and start fresh
python cli_ingestion.py rebuild
```

### 3. Query the System

```bash
python main.py "What are common Ayurvedic treatments for digestive issues?"
```

Output includes sparse/dense result counts, retrieved chunks with scores, the generated answer with inline citations, and citation metadata.

### 4. Multi-Hop Queries

For complex questions requiring multiple retrieval rounds:

```bash
# Via CLI (add multi_hop=True in code)
# Via Web GUI: enable "Thinking Mode" toggle
```

### 5. Web API + GUI

```bash
# Build the React UI (first time only)
cd ui && npm install && npm run build && cd ..

# Start the API server
python api.py
```

Open `http://localhost:8000` for the React GUI or use the REST API directly:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/query` | RAG query (single-pass or multi-hop) |
| `GET` | `/api/chunks` | Paginated document list |
| `GET` | `/api/chunks/{id}` | Get chunk with resolved children |
| `GET` | `/api/pages/{doc_id}` | Get full document context |
| `GET` | `/api/documents` | List ingested documents |
| `POST` | `/api/ingest/file` | Upload a single file |
| `POST` | `/api/ingest/files` | Upload multiple files |
| `POST` | `/api/ingest/text` | Ingest raw text |
| `DELETE` | `/api/documents/{doc_id}` | Delete a document |
| `DELETE` | `/api/documents` | Delete all documents |

## Qdrant Modes

Qdrant runs in one of two modes, controlled by the `QDRANT_URL` env var:

| Mode | Config | Qdrant location |
|------|--------|-----------------|
| **Remote** (Docker) | `QDRANT_URL=http://localhost:6333` | Docker container via `docker compose up -d` |
| **Local** (disk) | `QDRANT_URL` unset | Embedded local storage at `data/qdrant/` |

## Embedding Models

| Model | Use | Dim |
|-------|-----|-----|
| `BAAI/bge-small-en-v1.5` | Dense retrieval (fastembed, ONNX Runtime) | 384 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking (sentence-transformers, PyTorch) | — |

GPU acceleration is auto-detected: CUDA → DirectML → CPU fallback.

## Dependencies

| Package | Purpose |
|---------|---------|
| `qdrant-client` | Vector database for dense retrieval |
| `fastembed` | Embedding model inference (bge-small-en-v1.5) |
| `rank-bm25` | Legacy BM25 sparse retrieval |
| `sentence-transformers` | Cross-encoder reranking model |
| `pypdf` | PDF text extraction |
| `requests` | LLM API calls and Qdrant snapshot API |
| `python-dotenv` | Environment variable loading |
| `fastapi` / `uvicorn` | Web API server |
| `python-multipart` | File upload support |
| `onnxruntime-gpu` | GPU-accelerated embedding inference |
