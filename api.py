import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger("rag-api")

_sparse_retriever = None
_dense_retriever = None
_db = None
_loading = False
_load_error = None
_active_jobs: dict[str, dict] = {}


def _do_load():
    global _sparse_retriever, _dense_retriever, _db, _loading, _load_error
    _loading = True
    try:
        from indexing import SparseRetriever, DenseRetriever
        from sparse_fts import SparseFTS5Retriever
        from db import ChunkStoreDB

        fts_db_path = Path("data/sparse_fts.db")
        sparse_shards_dir = Path("data/sparse_shards")
        sparse_index_path = Path("data/sparse_index.pkl")

        logger.info("Loading sparse index...")
        _sparse_retriever = SparseFTS5Retriever.load(str(fts_db_path))
        if _sparse_retriever is not None:
            logger.info("Sparse: FTS5 loaded (%d documents)", _sparse_retriever.count())
        elif sparse_shards_dir.exists() and list(sparse_shards_dir.glob("shard_*.pkl")):
            _sparse_retriever = SparseRetriever.load_sharded(str(sparse_shards_dir))
            logger.info("Sparse: %d legacy shards loaded", len(_sparse_retriever.shards))
        elif sparse_index_path.exists():
            _sparse_retriever = SparseRetriever.load(str(sparse_index_path))
            logger.info("Sparse: single legacy index loaded")
        else:
            raise RuntimeError("No sparse index found. Run index_data.py or ingest data first.")

        logger.info("Loading dense index (fastembed model + Qdrant)...")
        _dense_retriever = DenseRetriever.load()
        logger.info("Dense: loaded")

        _db = ChunkStoreDB("data/chunks.db")
        logger.info("Chunk store: loaded")
        _loading = False
    except Exception as e:
        _load_error = str(e)
        _loading = False
        logger.error("Failed to load indices: %s", e)
        raise


@asynccontextmanager
async def lifespan(app):
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do_load)
    yield


app = FastAPI(title="Ayurveda RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    model: str | None = "gemma-4-31B-it"
    temperature: float | None = 0.2
    chat_history: list[Message] = []
    multi_hop: bool = False
    max_hops: int | None = 3


class Citation(BaseModel):
    citation_id: str
    source_id: str | None
    section_id: str | None
    supporting_child_ids: list[str]


class HopTrace(BaseModel):
    hop_number: int
    sub_queries: list[str]
    retrieved_section_ids: list[str]
    action: str


class ChunkResult(BaseModel):
    chunk_id: str
    score: float
    rerank_score: float
    text: str
    child_ids: list[str]
    doc_id: str | None = None
    chunk_type: str | None = None
    title: str | None = None
    source_url: str | None = None
    parent_id: str | None = None


class QueryResponse(BaseModel):
    answer_text: str | None
    citations: list[Citation]
    grounded: bool
    abstained: bool
    reason: str | None
    chunks: list[ChunkResult]
    hop_trace: list[HopTrace] | None = None


class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    progress: float
    errors: list[str]


class DocumentResponse(BaseModel):
    doc_id: str
    title: str
    source_type: str
    source_path: str | None
    file_name: str | None
    chunk_count: int
    created_at: str | None


@app.get("/health")
def health():
    if _db is not None:
        return {"status": "ok", "indices_loaded": True}
    if _loading:
        return {"status": "starting", "indices_loaded": False}
    return {"status": "error", "indices_loaded": False, "error": _load_error}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if _db is None:
        raise HTTPException(
            status_code=503,
            detail="Indices still loading. Try again shortly.",
        )

    from retrieval import hybrid_retrieve_with_rerank
    from generation import AnswerGenerator

    history_dicts = [{"role": m.role, "content": m.content} for m in request.chat_history]

    if request.multi_hop:
        from multi_hop import MultiHopOrchestrator

        orchestrator = MultiHopOrchestrator({
            "model": request.model,
            "temperature": request.temperature,
            "max_hops": request.max_hops or 3,
        })
        hop_trace, context_blocks, accumulated_sections = orchestrator.run(
            request.query,
            _sparse_retriever,
            _dense_retriever,
            _db,
            chat_history=history_dicts,
        )

        try:
            generator = AnswerGenerator({
                "model": request.model,
                "temperature": request.temperature,
            })
            answer = generator.generate(request.query, context_blocks, chat_history=history_dicts)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

        chunks = [
            ChunkResult(
                chunk_id=r.get("chunk_id", ""),
                score=r.get("rerank_score", r.get("score", 0.0)),
                rerank_score=r.get("rerank_score", r.get("score", 0.0)),
                text=r.get("text", ""),
                child_ids=r.get("child_ids", []),
                doc_id=r.get("doc_id"),
                chunk_type=r.get("chunk_type"),
                title=r.get("title"),
                source_url=r.get("source_url"),
                parent_id=r.get("parent_id"),
            )
            for r in accumulated_sections
        ]

        citations = [
            Citation(
                citation_id=c.get("citation_id", ""),
                source_id=c.get("source_id"),
                section_id=c.get("section_id"),
                supporting_child_ids=c.get("supporting_child_ids", []),
            )
            for c in answer.get("citations", [])
        ]

        trace = [
            HopTrace(
                hop_number=h["hop_number"],
                sub_queries=h["sub_queries"],
                retrieved_section_ids=h["retrieved_section_ids"],
                action=h["action"],
            )
            for h in hop_trace
        ]

        return QueryResponse(
            answer_text=answer.get("answer_text"),
            citations=citations,
            grounded=answer.get("grounded", True),
            abstained=answer.get("abstained", False),
            reason=answer.get("reason"),
            chunks=chunks,
            hop_trace=trace,
        )

    result = hybrid_retrieve_with_rerank(
        request.query,
        _sparse_retriever,
        _dense_retriever,
        _db,
    )

    context_blocks = result["results"]

    try:
        generator = AnswerGenerator({
            "model": request.model,
            "temperature": request.temperature,
        })
        answer = generator.generate(request.query, context_blocks, chat_history=history_dicts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

    chunks = [
        ChunkResult(
            chunk_id=r.get("chunk_id", ""),
            score=r.get("score", 0.0),
            rerank_score=r.get("rerank_score", r.get("score", 0.0)),
            text=r.get("text", ""),
            child_ids=r.get("child_ids", []),
            doc_id=r.get("doc_id"),
            chunk_type=r.get("chunk_type"),
            title=r.get("title"),
            source_url=r.get("source_url"),
            parent_id=r.get("parent_id"),
        )
        for r in result["results"]
    ]

    citations = [
        Citation(
            citation_id=c.get("citation_id", ""),
            source_id=c.get("source_id"),
            section_id=c.get("section_id"),
            supporting_child_ids=c.get("supporting_child_ids", []),
        )
        for c in answer.get("citations", [])
    ]

    return QueryResponse(
        answer_text=answer.get("answer_text"),
        citations=citations,
        grounded=answer.get("grounded", True),
        abstained=answer.get("abstained", False),
        reason=answer.get("reason"),
        chunks=chunks,
    )


_doc_count_cache = None


@app.get("/api/chunks")
def get_root_chunks(limit: int = 50, offset: int = 0):
    global _doc_count_cache
    if _db is None:
        raise HTTPException(status_code=503, detail="Indices still loading")

    if _doc_count_cache is None:
        _doc_count_cache = _db.count_children("document")

    chunks = _db.get_children_by_type("document", limit=limit, offset=offset)
    return {"chunks": chunks, "total": _doc_count_cache}


@app.get("/api/chunks/{chunk_id}")
def get_chunk(chunk_id: str):
    if _db is None:
        raise HTTPException(status_code=503, detail="Indices still loading")

    chunk = _db.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    children = []
    for child_id in chunk.get("children_ids", []):
        child = _db.get_chunk(child_id)
        if child:
            children.append(child)

    return {"chunk": chunk, "children": children}


@app.get("/api/pages/{doc_id}")
def get_page(doc_id: str):
    if _db is None:
        raise HTTPException(status_code=503, detail="Indices still loading")

    doc_chunks = _db.get_chunks_by_doc_id(doc_id, chunk_type="document")
    if not doc_chunks:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"page": doc_chunks[0]}


@app.get("/api/documents")
def get_documents(limit: int = 50, offset: int = 0):
    if _db is None:
        raise HTTPException(status_code=503, detail="Indices still loading")

    docs = _db.get_all_documents(limit=limit, offset=offset)
    total = _db.count_documents()
    return {"documents": docs, "total": total}


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename or "uploaded_file"

    from ingestion.connectors import WebUploadConnector
    connector = WebUploadConnector([(filename, content, file.content_type or "application/octet-stream")])
    return _run_ingestion(connector)


@app.post("/api/ingest/files")
async def ingest_files(files: list[UploadFile] = File(...)):
    file_data = [(f.filename or "uploaded", await f.read(), f.content_type or "application/octet-stream") for f in files]

    from ingestion.connectors import WebUploadConnector
    connector = WebUploadConnector(file_data)
    return _run_ingestion(connector)


class IngestTextRequest(BaseModel):
    text: str
    title: str = "Pasted Text"


@app.post("/api/ingest/text")
async def ingest_text(req: IngestTextRequest):
    from ingestion.connectors import RawTextConnector
    connector = RawTextConnector(req.text, req.title)
    return _run_ingestion(connector)


@app.get("/api/ingest/status/{job_id}")
def get_ingestion_status(job_id: str):
    job = _active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    if _db is None:
        raise HTTPException(status_code=503, detail="Indices still loading")

    doc = _db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = _db.get_chunks_by_doc_id(doc_id)
    for c in chunks:
        _db.conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (c["chunk_id"],))
    _db.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    _db.commit()

    global _doc_count_cache
    _doc_count_cache = None

    return {"status": "deleted", "doc_id": doc_id, "chunks_removed": len(chunks)}


@app.delete("/api/documents")
def delete_all_documents():
    if _db is None:
        raise HTTPException(status_code=503, detail="Indices still loading")

    _db.conn.execute("DELETE FROM chunks")
    _db.conn.execute("DELETE FROM documents")
    _db.commit()

    global _doc_count_cache
    _doc_count_cache = None

    return {"status": "all_documents_deleted"}


def _run_ingestion(connector) -> IngestionJobResponse:
    from ingestion.pipeline import IngestionPipeline, PipelineConfig
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    job_id = str(uuid.uuid4())
    _active_jobs[job_id] = {"job_id": job_id, "status": "starting", "message": "Starting ingestion...",
                            "progress": 0.0, "errors": []}

    config = PipelineConfig(storage_dir="data")
    pipeline = IngestionPipeline(config)

    job = pipeline.run(connector)
    resp = IngestionJobResponse(
        job_id=job.job_id,
        status=job.status,
        message=job.message,
        progress=job.progress,
        errors=job.errors,
    )
    _active_jobs[job_id] = resp.model_dump()

    global _doc_count_cache
    _doc_count_cache = None

    return resp


import os
if os.path.exists("ui/dist"):
    app.mount("/", StaticFiles(directory="ui/dist", html=True), name="ui")
else:
    logger.warning("ui/dist not found. The web GUI will not be served.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
