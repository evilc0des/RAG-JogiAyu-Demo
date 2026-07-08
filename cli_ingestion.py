import argparse
import os
import sys
from pathlib import Path

from ingestion.pipeline import IngestionPipeline, PipelineConfig
from ingestion.connectors import FileConnector, DirectoryConnector, RawTextConnector
from ingestion.file_readers import SUPPORTED_EXTENSIONS


def cmd_ingest(args):
    config = PipelineConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        split_on_paragraphs=not args.no_paragraph_split,
        use_semantic_splitting=not args.no_semantic,
        use_llm_metadata=args.llm_metadata,
        storage_dir=args.storage_dir,
    )

    pipeline = IngestionPipeline(config)

    if args.text:
        connector = RawTextConnector(args.text, title=args.title or "CLI Input")
    elif args.file:
        path = Path(args.file)
        if path.is_dir():
            connector = DirectoryConnector(path, args.extensions)
        else:
            connector = FileConnector(path)
    elif args.dir:
        connector = DirectoryConnector(Path(args.dir), args.extensions)
    else:
        print("Error: Must provide --file, --dir, or --text", file=sys.stderr)
        sys.exit(1)

    print(f"Starting ingestion...")
    job = pipeline.run(connector)

    print(f"Status: {job.status}")
    print(f"Message: {job.message}")
    if job.errors:
        print(f"Errors: {job.errors}")

    if job.status == "completed":
        print("Ingestion complete.")
    else:
        sys.exit(1)


def cmd_status(args):
    from db import ChunkStoreDB
    db_path = Path(args.storage_dir) / "chunks.db"
    if not db_path.exists():
        print("No database found. Run ingestion first.")
        return

    db = ChunkStoreDB(str(db_path))
    doc_count = db.count_children("document")
    chunk_count = db.count_children("chunk")
    docs = db.get_all_documents(limit=100)

    print(f"Database: {db_path}")
    print(f"Documents: {doc_count}")
    print(f"Chunks: {chunk_count}")
    print(f"\nRecent documents:")
    for doc in docs:
        print(f"  [{doc['source_type']}] {doc['title']} ({doc['chunk_count']} chunks) - {doc['created_at']}")
    db.close()


def cmd_list(args):
    from db import ChunkStoreDB
    db_path = Path(args.storage_dir) / "chunks.db"
    if not db_path.exists():
        print("No database found. Run ingestion first.")
        return

    db = ChunkStoreDB(str(db_path))
    docs = db.get_all_documents(limit=args.limit, offset=args.offset)
    for doc in docs:
        print(f"{doc['doc_id']} | [{doc['source_type']}] {doc['title']} | {doc['chunk_count']} chunks | {doc['created_at']}")
    db.close()


def cmd_delete(args):
    from db import ChunkStoreDB
    db_path = Path(args.storage_dir) / "chunks.db"
    if not db_path.exists():
        print("No database found.")
        return

    db = ChunkStoreDB(str(db_path))
    doc = db.get_document(args.doc_id)
    if not doc:
        print(f"Document {args.doc_id} not found.")
        db.close()
        return

    chunks = db.get_chunks_by_doc_id(args.doc_id)
    for c in chunks:
        db.conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (c["chunk_id"],))
    db.conn.execute("DELETE FROM documents WHERE doc_id = ?", (args.doc_id,))
    db.commit()
    print(f"Deleted document '{doc['title']}' and {len(chunks)} chunk(s).")
    db.close()


def cmd_rebuild(args):
    for suffix in ("", "-shm", "-wal"):
        p = Path(args.storage_dir) / f"chunks.db{suffix}"
        if p.exists():
            p.unlink()
    p = Path(args.storage_dir) / "sparse_fts.db"
    if p.exists():
        p.unlink()
        for s in ("-shm", "-wal"):
            ps = Path(args.storage_dir) / f"sparse_fts.db{s}"
            if ps.exists():
                ps.unlink()
    qdrant = Path(args.storage_dir) / "qdrant"
    if qdrant.exists():
        import shutil
        shutil.rmtree(qdrant)
    sparse = Path(args.storage_dir) / "sparse_shards"
    if sparse.exists():
        import shutil
        shutil.rmtree(sparse)

    qdrant_url = os.environ.get("QDRANT_URL")
    if qdrant_url:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url)
        if client.collection_exists("dense_index"):
            client.delete_collection("dense_index")
        client.close()

    print("All indices and databases cleared. Ready for fresh ingestion.")


def main():
    parser = argparse.ArgumentParser(description="Ayurveda RAG - CLI Data Ingestion")
    sub = parser.add_subparsers(dest="command", help="Commands")

    p_ingest = sub.add_parser("ingest", help="Ingest documents into the RAG system")
    p_ingest.add_argument("--file", type=str, help="Path to a file to ingest")
    p_ingest.add_argument("--dir", type=str, help="Path to a directory of files to ingest")
    p_ingest.add_argument("--text", type=str, help="Raw text to ingest directly")
    p_ingest.add_argument("--title", type=str, help="Title for raw text input")
    p_ingest.add_argument("--extensions", type=str, nargs="*", help="File extensions to include (dir mode)")
    p_ingest.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens (default: 512)")
    p_ingest.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap in tokens (default: 64)")
    p_ingest.add_argument("--no-paragraph-split", action="store_true", help="Disable paragraph-aware splitting")
    p_ingest.add_argument("--no-semantic", action="store_true", help="Disable semantic splitting")
    p_ingest.add_argument("--llm-metadata", action="store_true", help="Enable LLM-generated metadata")
    p_ingest.add_argument("--storage-dir", type=str, default="data", help="Storage directory (default: data)")

    p_status = sub.add_parser("status", help="Show ingestion status and statistics")
    p_status.add_argument("--storage-dir", type=str, default="data")

    p_list = sub.add_parser("list", help="List ingested documents")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--offset", type=int, default=0)
    p_list.add_argument("--storage-dir", type=str, default="data")

    p_delete = sub.add_parser("delete", help="Delete a document and its chunks")
    p_delete.add_argument("doc_id", type=str, help="Document ID to delete")
    p_delete.add_argument("--storage-dir", type=str, default="data")

    p_rebuild = sub.add_parser("rebuild", help="Delete all indices and start fresh")
    p_rebuild.add_argument("--storage-dir", type=str, default="data")

    args = parser.parse_args()

    commands = {
        "ingest": cmd_ingest,
        "status": cmd_status,
        "list": cmd_list,
        "delete": cmd_delete,
        "rebuild": cmd_rebuild,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
