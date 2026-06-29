import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path, override=True)


def _sha256_hex(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 * 1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def _upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, max_retries=3):
    last_exc = None
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                msg = str(exc).split("\n")[0]
                print(f"  Retry {attempt + 1}/{max_retries} in {wait}s: {msg}")
                time.sleep(wait)
    raise last_exc


def _upload_file_sharded(api, file_path, repo_id, remote_dir, part_size_mb, tmp_root):
    part_size = part_size_mb * 1024 * 1024
    file_size = file_path.stat().st_size
    parts = []
    fname = file_path.name

    tmpdir = tmp_root / f"shard_upload_{fname}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, "rb") as f:
            part_idx = 0
            while True:
                data = f.read(part_size)
                if not data:
                    break
                part_name = f"{fname}.part_{part_idx:03d}"
                tmp_part = tmpdir / part_name
                tmp_part.write_bytes(data)
                size_str = f"{len(data) / 1024 / 1024:.0f} MB" if len(data) < 1024 * 1024 * 1024 else f"{len(data) / 1024 / 1024 / 1024:.1f} GB"
                if part_idx > 0:
                    print(f"  [{part_idx + 1}] {part_name} ({size_str})")
                else:
                    print(f"  [1] {part_name} ({size_str})")
                _upload_with_retry(
                    api, str(tmp_part),
                    f"{remote_dir}/{part_name}",
                    repo_id, "dataset",
                )
                parts.append(part_name)
                part_idx += 1

        manifest = {
            "original_name": fname,
            "original_size": file_size,
            "sha256": _sha256_hex(file_path),
            "parts": parts,
        }
        manifest_name = f"{fname}.manifest.json"
        tmp_manifest = tmpdir / manifest_name
        tmp_manifest.write_text(json.dumps(manifest))
        _upload_with_retry(
            api, str(tmp_manifest),
            f"{remote_dir}/{manifest_name}",
            repo_id, "dataset",
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"  Uploaded {fname} ({file_size / 1024 / 1024 / 1024:.1f} GB) in {len(parts)} shards")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Package pre-built index data and upload to HuggingFace Hub"
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
    parser.add_argument("--sparse-shards", default="data/sparse_shards",
                        help="Path to legacy BM25 shards dir (fallback if no FTS5)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--shard-size-mb", type=int, default=2000,
                        help="Max size (MB) per uploaded shard (default: 2000)")
    parser.add_argument("--local-snapshot", default=None,
                        help="Path to an already-downloaded Qdrant snapshot (skips create+download)")
    parser.add_argument("--tmp-dir", default="data/tmp",
                        help="Directory for temp files during sharding (default: data/tmp)")
    args = parser.parse_args()

    tmp_root = Path(args.tmp_dir)
    tmp_root.mkdir(parents=True, exist_ok=True)

    repo_id = args.repo
    if not repo_id:
        print("ERROR: HF_DATASET_REPO not set. Provide --repo or set the env var.")
        sys.exit(1)

    token = args.token
    if not token:
        print("ERROR: HF_TOKEN not set. Provide --token or set the env var.")
        sys.exit(1)

    db_path = Path(args.db)
    fts_path = Path(args.sparse_fts)
    sparse_dir = Path(args.sparse_shards)

    if not db_path.exists():
        print(f"ERROR: chunks.db not found at {db_path}. Run index_data.py first.")
        sys.exit(1)

    use_fts = fts_path.exists()
    shard_files = sorted(sparse_dir.glob("shard_*.pkl")) if not use_fts else []
    if not use_fts and not shard_files:
        print(f"ERROR: no sparse index found. Expected {fts_path} or shards in {sparse_dir}.")
        print("Run index_data.py first.")
        sys.exit(1)

    if args.local_snapshot:
        snapshot_path = Path(args.local_snapshot)
        if not snapshot_path.exists():
            print(f"ERROR: --local-snapshot file not found: {snapshot_path}")
            sys.exit(1)
        print(f"Using local snapshot: {snapshot_path} "
              f"({snapshot_path.stat().st_size / 1024 / 1024:.1f} MB)")
        tmpdir = None
    else:
        from indexing import create_qdrant_snapshot, download_qdrant_snapshot

        qdrant_url = args.qdrant_url
        collection_name = args.collection

        print(f"Creating Qdrant snapshot for collection '{collection_name}' at {qdrant_url} ...")
        result = create_qdrant_snapshot(qdrant_url, collection_name, api_key=args.qdrant_api_key)
        snapshot_name = result["name"]
        print(f"  Snapshot created: {snapshot_name}")

        tmpdir = Path("data/tmp/qdrant_snapshot")
        tmpdir.mkdir(parents=True, exist_ok=True)
        snapshot_path = tmpdir / snapshot_name
        print(f"Downloading snapshot to {snapshot_path} ...")
        download_qdrant_snapshot(qdrant_url, collection_name, snapshot_name,
                                 str(snapshot_path), api_key=args.qdrant_api_key)
        print(f"  Downloaded ({snapshot_path.stat().st_size / 1024 / 1024:.1f} MB)")

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    try:
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    except Exception:
        pass

    db_size_mb = db_path.stat().st_size / 1024 / 1024
    if db_size_mb > args.shard_size_mb:
        print(f"Uploading chunks.db ({db_size_mb:.1f} MB) in shards to {repo_id} ...")
        _upload_file_sharded(api, db_path, repo_id, "chunks_shards", args.shard_size_mb, tmp_root)
    else:
        print(f"Uploading chunks.db ({db_size_mb:.1f} MB) to {repo_id} ...")
        api.upload_file(
            path_or_fileobj=str(db_path),
            path_in_repo="chunks.db",
            repo_id=repo_id,
            repo_type="dataset",
        )

    if use_fts:
        fts_size_mb = fts_path.stat().st_size / 1024 / 1024
        if fts_size_mb > args.shard_size_mb:
            print(f"Uploading sparse_fts.db ({fts_size_mb:.1f} MB) in shards to {repo_id} ...")
            _upload_file_sharded(api, fts_path, repo_id, "sparse_fts_shards", args.shard_size_mb, tmp_root)
        else:
            print(f"Uploading sparse_fts.db ({fts_size_mb:.1f} MB) to {repo_id} ...")
            api.upload_file(
                path_or_fileobj=str(fts_path),
                path_in_repo="sparse_fts.db",
                repo_id=repo_id,
                repo_type="dataset",
            )
    else:
        print(f"Uploading {len(shard_files)} sparse shards to {repo_id}/sparse_shards/ ...")
        for i, sf in enumerate(shard_files):
            api.upload_file(
                path_or_fileobj=str(sf),
                path_in_repo=f"sparse_shards/{sf.name}",
                repo_id=repo_id,
                repo_type="dataset",
            )
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(shard_files)} shards uploaded")

    snapshot_size_mb = snapshot_path.stat().st_size / 1024 / 1024
    if snapshot_size_mb > args.shard_size_mb:
        print(f"Uploading Qdrant snapshot ({snapshot_size_mb:.1f} MB) in shards to {repo_id} ...")
        _upload_file_sharded(api, snapshot_path, repo_id, "dense_index_shards", args.shard_size_mb, tmp_root)
    else:
        print(f"Uploading Qdrant snapshot ({snapshot_size_mb:.1f} MB) to {repo_id} ...")
        api.upload_file(
            path_or_fileobj=str(snapshot_path),
            path_in_repo="dense_index.snapshot",
            repo_id=repo_id,
            repo_type="dataset",
        )

    sparse_desc = "sparse_fts.db" if use_fts else f"sparse_shards/ ({len(shard_files)} shards)"
    print(f"\nPackage uploaded successfully to {repo_id}")
    print(f"  Files: chunks.db, {sparse_desc}, dense_index.snapshot")

    if tmpdir is not None:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
