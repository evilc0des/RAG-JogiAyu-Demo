import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .connectors import BaseConnector, IngestionJob
from .file_readers import ParsedDocument


@dataclass
class PipelineConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    split_on_paragraphs: bool = True
    use_semantic_splitting: bool = True
    use_llm_metadata: bool = False
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    storage_dir: str = "data"


class IngestionPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def run(self, connector: BaseConnector) -> IngestionJob:
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        job_id = str(uuid.uuid4())
        job = IngestionJob(job_id=job_id, status="starting", message="Reading source files...")

        documents = connector.ingest()
        if hasattr(connector, "errors") and connector.errors:
            job.errors = connector.errors
            if not documents:
                job.status = "failed"
                return job

        job.documents = documents
        job.status = "chunking"
        job.message = f"Chunking {len(documents)} document(s)..."
        job.progress = 0.2

        from chunking import HybridChunker
        chunker = HybridChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            split_on_paragraphs=self.config.split_on_paragraphs,
            use_semantic=self.config.use_semantic_splitting,
            llm_metadata=self.config.use_llm_metadata,
        )

        from db import ChunkStoreDB
        db_path = str(Path(self.config.storage_dir) / "chunks.db")
        db = ChunkStoreDB(db_path)

        try:
            chunker.chunk_documents(documents, db)
            db.commit()
        except Exception as e:
            job.status = "failed"
            job.message = f"Chunking failed: {e}"
            job.errors.append(str(e))
            db.close()
            return job
        finally:
            pass

        doc_count = db.count_children("document")
        chunk_count = db.count_children("chunk")
        db.close()

        job.status = "indexing"
        job.message = f"Building sparse and dense indexes for {chunk_count} chunks..."
        job.progress = 0.6

        try:
            self._build_indexes(job)
        except Exception as e:
            job.status = "partial"
            job.message = f"Indexing incomplete: {e}. Chunks are available but search may be limited."
            job.errors.append(str(e))
            return job

        job.status = "completed"
        job.message = f"Ingested {doc_count} document(s), {chunk_count} chunks"
        job.progress = 1.0
        return job

    def _build_indexes(self, job: IngestionJob):
        from sparse_fts import SparseFTS5Retriever
        from indexing import build_dense_index_from_db
        import os

        storage = Path(self.config.storage_dir)
        db_path = str(storage / "chunks.db")
        fts_path = str(storage / "sparse_fts.db")
        dense_path = str(storage / "qdrant")
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None

        job.message = "Building FTS5 sparse index..."
        fts = SparseFTS5Retriever(fts_path)
        fts.build_from_db(db_path)
        fts.close()

        job.message = "Building dense (Qdrant) index..."
        build_dense_index_from_db(db_path, dense_path, batch_size=1000,
                                  qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)
