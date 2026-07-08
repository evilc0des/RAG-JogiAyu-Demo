import argparse
import os
import shutil
from pathlib import Path

from ingestion.pipeline import IngestionPipeline, PipelineConfig
from ingestion.connectors import DirectoryConnector, FileConnector
from db import ChunkStoreDB


def main():
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None

    parser = argparse.ArgumentParser(description="Ayurveda RAG - Build indices from ingested documents")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Directory containing source files to ingest (default: data/raw)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete existing data and re-index from scratch")
    parser.add_argument("--skip-chunking", action="store_true",
                        help="Skip chunking step (use when chunks.db is already populated)")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap in tokens")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic splitting")
    parser.add_argument("--legacy-bm25", action="store_true",
                        help="Build old-style rank_bm25 pickle shards instead of FTS5")
    args = parser.parse_args()

    DB_PATH = "data/chunks.db"
    SPARSE_SHARDS_DIR = "data/sparse_shards"
    FTS_DB_PATH = "data/sparse_fts.db"
    DENSE_PATH = "data/qdrant"

    if args.rebuild:
        print("Rebuild requested - deleting existing data...")
        for suffix in ("", "-shm", "-wal"):
            p = Path(DB_PATH + suffix)
            if p.exists():
                p.unlink()
                print(f"  Deleted {p}")
        if Path(SPARSE_SHARDS_DIR).exists():
            shutil.rmtree(SPARSE_SHARDS_DIR)
            print(f"  Deleted {SPARSE_SHARDS_DIR}")
        for suffix in ("", "-shm", "-wal"):
            p = Path(FTS_DB_PATH + suffix)
            if p.exists():
                p.unlink()
                print(f"  Deleted {p}")
        if qdrant_url:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            if client.collection_exists("dense_index"):
                client.delete_collection("dense_index")
                print("  Deleted dense_index collection from Qdrant")
            client.close()
        elif Path(DENSE_PATH).exists():
            shutil.rmtree(DENSE_PATH)
            print(f"  Deleted {DENSE_PATH}")

    Path(SPARSE_SHARDS_DIR).mkdir(parents=True, exist_ok=True)

    db = ChunkStoreDB(DB_PATH)
    doc_count = db.count_children("document")

    if args.skip_chunking:
        print(f"Skipping chunking. chunks.db has {doc_count} documents, {db.count_children('chunk')} chunks.")
        db.close()
    else:
        db.close()
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Data directory '{args.data_dir}' not found. Creating it...")
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Place source files (transcripts, books) in '{args.data_dir}' and re-run.")
            return

        files = list(data_dir.rglob("*"))
        if not any(f.is_file() for f in files):
            print(f"No files found in '{args.data_dir}'. Place source files there and re-run.")
            return

        print(f"Ingesting files from '{args.data_dir}'...")

        config = PipelineConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            split_on_paragraphs=True,
            use_semantic_splitting=not args.no_semantic,
            use_llm_metadata=False,
            storage_dir="data",
        )

        pipeline = IngestionPipeline(config)
        connector = DirectoryConnector(data_dir)
        job = pipeline.run(connector)

        print(f"Ingestion status: {job.status}")
        print(f"Message: {job.message}")
        if job.errors:
            print(f"Errors: {job.errors}")

        if job.status != "completed":
            print("Ingestion did not complete successfully. Skipping index build.")
            return

    from indexing import build_sparse_indexes_from_db, build_dense_index_from_db
    from sparse_fts import SparseFTS5Retriever

    if args.legacy_bm25:
        build_sparse_indexes_from_db(DB_PATH, SPARSE_SHARDS_DIR, shard_size=100000)
        print("Sparse (BM25 sharded) indexes built.")
    else:
        print("Building FTS5 sparse index...")
        fts = SparseFTS5Retriever(FTS_DB_PATH)
        fts.build_from_db(DB_PATH)
        fts.close()
        print("Sparse (FTS5) index built.")

    build_dense_index_from_db(DB_PATH, DENSE_PATH, batch_size=1000,
                              qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)
    print("Dense (Qdrant) index built.")
    print("Indexing complete.")


if __name__ == "__main__":
    main()
