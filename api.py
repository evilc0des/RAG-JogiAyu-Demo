import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("rag-api")

_sparse_retriever = None
_dense_retriever = None
_db = None
_loading = False
_load_error = None


def _do_load():
    """Load all retrievers and the chunk store DB. Called once at startup."""
    global _sparse_retriever, _dense_retriever, _db, _loading, _load_error
    _loading = True
    try:
        from indexing import SparseRetriever, DenseRetriever
        from sparse_fts import SparseFTS5Retriever
        from db import ChunkStoreDB

        fts_db_path = Path("data/sparse_fts.db")
        sparse_shards_dir = Path("data/sparse_shards")
        sparse_index_path = Path("data/sparse_index.pkl")

        # Prefer FTS5 (disk-backed, near-zero RAM) over legacy pickle shards
        logger.info("Loading sparse index...")
        _sparse_retriever = SparseFTS5Retriever.load(str(fts_db_path))
        if _sparse_retriever is not None:
            logger.info("Sparse: FTS5 loaded (%d children)", _sparse_retriever.count())
        elif sparse_shards_dir.exists() and list(sparse_shards_dir.glob("shard_*.pkl")):
            _sparse_retriever = SparseRetriever.load_sharded(str(sparse_shards_dir))
            logger.info("Sparse: %d legacy shards loaded", len(_sparse_retriever.shards))
        elif sparse_index_path.exists():
            _sparse_retriever = SparseRetriever.load(str(sparse_index_path))
            logger.info("Sparse: single legacy index loaded")
        else:
            raise RuntimeError("No sparse index found. Run index_data.py first.")

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
    """Preload all indices at startup, before accepting requests."""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do_load)
    yield


app = FastAPI(title="RAG Demo API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    model: str | None = "gemma-4-31B-it"
    temperature: float | None = 0.2


class Citation(BaseModel):
    citation_id: str
    source_id: str | None
    section_id: str | None
    supporting_child_ids: list[str]


class SectionResult(BaseModel):
    chunk_id: str
    score: float
    rerank_score: float
    text: str
    child_ids: list[str]


class QueryResponse(BaseModel):
    answer_text: str | None
    citations: list[Citation]
    grounded: bool
    abstained: bool
    reason: str | None
    sections: list[SectionResult]


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
    from generation import build_context_blocks, AnswerGenerator

    result = hybrid_retrieve_with_rerank(
        request.query,
        _sparse_retriever,
        _dense_retriever,
        _db,
    )

    context_blocks = build_context_blocks(result["results"])

    try:
        generator = AnswerGenerator({
            "model": request.model,
            "temperature": request.temperature,
        })
        answer = generator.generate(request.query, context_blocks)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

    sections = [
        SectionResult(
            chunk_id=r.get("chunk_id", ""),
            score=r.get("score", 0.0),
            rerank_score=r.get("rerank_score", r.get("score", 0.0)),
            text=r.get("text", "")[:500],
            child_ids=r.get("child_ids", []),
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
        sections=sections,
    )

