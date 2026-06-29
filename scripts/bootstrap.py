import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path, override=True)


def _download_sharded_file(repo_id, filename, shard_dir, local_path, token, repo_files, tmp_root):
    manifest_key = f"{shard_dir}/{filename}.manifest.json"
    if manifest_key not in repo_files:
        return False

    from huggingface_hub import hf_hub_download

    print(f"Downloading {filename} manifest ...")
    hf_hub_download(
        repo_id=repo_id, filename=manifest_key, repo_type="dataset",
        token=token, local_dir="data", local_dir_use_symlinks=False,
    )
    manifest_path = Path("data") / manifest_key
    manifest = json.loads(manifest_path.read_text())
    manifest_path.unlink()

    part_dir = Path("data") / shard_dir
    tmpdir = tmp_root / f"shard_dl_{filename}"
    tmpdir.mkdir(parents=True, exist_ok=True)

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256()
        print(f"Downloading and reassembling {len(manifest['parts'])} shards for {filename} ...")
        with open(local_path, "wb") as out:
            for i, part_name in enumerate(manifest["parts"]):
                part_remote = f"{shard_dir}/{part_name}"
                hf_hub_download(
                    repo_id=repo_id, filename=part_remote, repo_type="dataset",
                    token=token, local_dir=str(tmpdir), local_dir_use_symlinks=False,
                )
                part_path = tmpdir / part_remote
                data = part_path.read_bytes()
                out.write(data)
                sha.update(data)
                part_path.unlink()
                if (i + 1) % 5 == 0 or i == len(manifest["parts"]) - 1:
                    print(f"  {i + 1}/{len(manifest['parts'])} shards")

        digest = sha.hexdigest()
        if digest != manifest["sha256"]:
            local_path.unlink(missing_ok=True)
            print(f"ERROR: SHA256 mismatch for {filename} after reassembly.")
            print(f"  Expected: {manifest['sha256']}")
            print(f"  Got:      {digest}")
            sys.exit(1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(part_dir, ignore_errors=True)

    print(f"  Reassembled {filename} ({local_path.stat().st_size / 1024 / 1024:.1f} MB) from {len(manifest['parts'])} parts")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap data from HuggingFace Hub for RAG pipeline"
    )
    parser.add_argument("--collection", default="dense_index",
                        help="Qdrant collection name (default: dense_index)")
    parser.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO"),
                        help="HuggingFace dataset repo ID (e.g. your-org/rag-demo-data)")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT_API_KEY") or None)
    parser.add_argument("--db", default="data/chunks.db")
    parser.add_argument("--sparse-fts", default="data/sparse_fts.db",
                        help="Path to FTS5 sparse index (default: data/sparse_fts.db)")
    parser.add_argument("--sparse-shards", default="data/sparse_shards")
    parser.add_argument("--snapshot", default="data/dense_index.snapshot")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-import even if data exists")
    parser.add_argument("--download-only", action="store_true",
                        help="Download data files but skip Qdrant snapshot import")
    parser.add_argument("--tmp-dir", default="data/tmp",
                        help="Directory for temp files during shard download (default: data/tmp)")
    args = parser.parse_args()

    tmp_root = Path(args.tmp_dir)
    tmp_root.mkdir(parents=True, exist_ok=True)

    repo_id = args.repo
    if not repo_id:
        print("ERROR: HF_DATASET_REPO not set. Provide --repo or set the env var.")
        sys.exit(1)

    token = args.token
    if not token:
        print("WARNING: HF_TOKEN not set. Attempting anonymous download (may fail for private repos).")

    db_path = Path(args.db)
    fts_path = Path(args.sparse_fts)
    sparse_dir = Path(args.sparse_shards)
    snapshot_path = Path(args.snapshot)

    # --- Step 1: Download data files from HF ---

    has_sparse = (
        fts_path.exists()
        or (sparse_dir.exists() and list(sparse_dir.glob("shard_*.pkl")))
    )
    needs_download = args.force or not (
        db_path.exists() and has_sparse and snapshot_path.exists()
    )

    if needs_download:
        from huggingface_hub import hf_hub_download, list_repo_files

        print(f"Fetching file list from {repo_id} ...")
        try:
            repo_files = list_repo_files(repo_id, repo_type="dataset", token=token)
        except Exception as e:
            print(f"ERROR: Failed to list repo files: {e}")
            sys.exit(1)

        print(f"  Found {len(repo_files)} files in repo")

        has_chunks = "chunks.db" in repo_files
        has_chunks_sharded = "chunks_shards/chunks.db.manifest.json" in repo_files
        has_snapshot = "dense_index.snapshot" in repo_files
        has_snapshot_sharded = "dense_index_shards/dense_index.snapshot.manifest.json" in repo_files
        has_fts = "sparse_fts.db" in repo_files
        has_fts_sharded = "sparse_fts_shards/sparse_fts.db.manifest.json" in repo_files
        sparse_files = [f for f in repo_files
                        if f.startswith("sparse_shards/") and f.endswith(".pkl")]

        print(f"  chunks.db: {'sharded' if has_chunks_sharded else 'yes' if has_chunks else 'MISSING'}")
        print(f"  dense_index.snapshot: {'sharded' if has_snapshot_sharded else 'yes' if has_snapshot else 'MISSING'}")
        print(f"  sparse_fts.db: {'sharded' if has_fts_sharded else 'yes' if has_fts else 'no'}")
        print(f"  sparse shards (legacy): {len(sparse_files)} files")

        if not has_chunks and not has_chunks_sharded:
            print("ERROR: chunks.db not found in repo.")
            sys.exit(1)

        if not has_fts and not has_fts_sharded and not sparse_files:
            print("ERROR: No sparse index found in repo (neither sparse_fts.db nor sparse_shards/).")
            sys.exit(1)

        # Download chunks.db (sharded or single file)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if has_chunks_sharded:
            _download_sharded_file(repo_id, "chunks.db", "chunks_shards", db_path, token, repo_files, tmp_root)
        else:
            print("Downloading chunks.db ...")
            hf_hub_download(
                repo_id=repo_id, filename="chunks.db", repo_type="dataset",
                token=token, local_dir="data", local_dir_use_symlinks=False,
            )
            print(f"  Saved to {db_path}")

        # Download sparse index (FTS5 preferred over legacy shards)
        if has_fts_sharded:
            _download_sharded_file(repo_id, "sparse_fts.db", "sparse_fts_shards", fts_path, token, repo_files, tmp_root)
        elif has_fts:
            print("Downloading sparse_fts.db ...")
            hf_hub_download(
                repo_id=repo_id, filename="sparse_fts.db", repo_type="dataset",
                token=token, local_dir="data", local_dir_use_symlinks=False,
            )
            print(f"  Saved to {fts_path}")
        elif sparse_files:
            sparse_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {len(sparse_files)} sparse shards (legacy) ...")
            for i, sf in enumerate(sparse_files):
                hf_hub_download(
                    repo_id=repo_id, filename=sf, repo_type="dataset",
                    token=token, local_dir="data", local_dir_use_symlinks=False,
                )
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(sparse_files)} shards downloaded")
            print(f"  {len(sparse_files)} sparse shards saved to {sparse_dir}")

        # Download dense index snapshot (sharded or single file)
        if has_snapshot_sharded:
            _download_sharded_file(repo_id, "dense_index.snapshot", "dense_index_shards", snapshot_path, token, repo_files, tmp_root)
        elif has_snapshot:
            print("Downloading dense_index.snapshot ...")
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=repo_id, filename="dense_index.snapshot", repo_type="dataset",
                token=token, local_dir="data", local_dir_use_symlinks=False,
            )
            print(f"  Saved to {snapshot_path}")
        else:
            print("ERROR: dense_index.snapshot not found in repo.")
            sys.exit(1)
    else:
        print(f"Data files already exist. Use --force to re-download.")

    if args.download_only:
        print("Download complete (--download-only). Skipping Qdrant snapshot import.")
        return

    # --- Step 2: Import snapshot into Docker Qdrant ---

    if not snapshot_path.exists():
        print(f"ERROR: Snapshot file not found at {snapshot_path}")
        sys.exit(1)

    qdrant_url = args.qdrant_url
    collection_name = args.collection

    from indexing import wait_for_qdrant, import_qdrant_snapshot

    print(f"Waiting for Qdrant at {qdrant_url} ...")
    try:
        wait_for_qdrant(qdrant_url)
        print("  Qdrant is healthy.")
    except TimeoutError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(url=qdrant_url, api_key=args.qdrant_api_key)

    if client.collection_exists(collection_name):
        point_count = client.count(collection_name=collection_name).count
        if point_count > 0 and not args.force:
            print(f"Collection '{collection_name}' already has {point_count} points. "
                  "Skipping snapshot import. Use --force to re-import.")
            client.close()
            return

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    client.close()

    print(f"Importing snapshot into Qdrant collection '{collection_name}' ...")
    try:
        import_qdrant_snapshot(
            qdrant_url, collection_name, str(snapshot_path),
            api_key=args.qdrant_api_key,
        )
        print("  Snapshot imported successfully.")
    except Exception as e:
        print(f"ERROR: Snapshot import failed: {e}")
        sys.exit(1)

    print("\nBootstrap complete. Data is ready.")


if __name__ == "__main__":
    main()
