#!/bin/bash
set -e

# --- Raise OS limits for Qdrant + RocksDB with large collections ---
# RocksDB opens many SST files; default limits cause "IO error: While open
# a file for random read" on the second query with millions of vectors.
ulimit -n 65536 2>/dev/null || echo "WARNING: could not raise open-file limit (ulimit -n)"
if [ -w /proc/sys/vm/max_map_count ]; then
    echo 262144 > /proc/sys/vm/max_map_count
else
    echo "NOTE: cannot set vm.max_map_count (not privileged). If Qdrant crashes, run the container with --privileged or set on the host."
fi

if [ ! -L /app/data ]; then
    ln -s /data /app/data
fi

mkdir -p /app/data/qdrant_storage

export QDRANT__STORAGE__STORAGE_PATH=/app/data/qdrant_storage

MODE="${1:-serve}"
shift || true

SNAPSHOT_ARG=""
START_QDRANT=true

# If QDRANT_URL points to a remote server (not localhost), skip starting Qdrant locally
if [ -n "$QDRANT_URL" ] && [ "$QDRANT_URL" != "http://localhost:6333" ]; then
    echo "QDRANT_URL=$QDRANT_URL is remote. Skipping local Qdrant start."
    START_QDRANT=false
fi

if [ "$START_QDRANT" = "true" ]; then
    if [ "$MODE" = "serve" ]; then
        if [ -n "$(ls -A /app/data/qdrant_storage 2>/dev/null)" ]; then
            echo "Qdrant storage has data from prior boot. Starting normally."
        elif [ -f /app/data/dense_index.snapshot ]; then
            echo "Snapshot found. Starting Qdrant from snapshot..."
            SNAPSHOT_ARG="--storage-snapshot /app/data/dense_index.snapshot"
        elif [ -n "$HF_DATASET_REPO" ]; then
            echo "Downloading data from HuggingFace Hub..."
            python3 scripts/bootstrap.py --download-only
            echo "Starting Qdrant from snapshot..."
            SNAPSHOT_ARG="--storage-snapshot /app/data/dense_index.snapshot"
        else
            echo "ERROR: No indices found and HF_DATASET_REPO not set."
            echo "Run with 'index' mode first, or set HF_DATASET_REPO + HF_TOKEN."
            exit 1
        fi
    fi

    echo "Starting Qdrant..."
    QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=2 /usr/local/bin/qdrant $SNAPSHOT_ARG &

    echo "Waiting for Qdrant health (large collections may take a few minutes)..."
    for i in $(seq 1 300); do
        if curl -sf http://localhost:6333/ > /dev/null 2>&1; then
            echo "Qdrant is healthy. (ready in ${i}s)"
            break
        fi
        if [ "$i" -eq 300 ]; then
            echo "ERROR: Qdrant failed to start within 300s"
            exit 1
        fi
        if [ $((i % 15)) -eq 0 ]; then
            echo "  Still waiting for Qdrant... (${i}s elapsed)"
        fi
        sleep 1
    done
else
    echo "Waiting for remote Qdrant at $QDRANT_URL..."
    for i in $(seq 1 300); do
        if curl -sf "$QDRANT_URL/" > /dev/null 2>&1; then
            echo "Remote Qdrant is healthy. (ready in ${i}s)"
            break
        fi
        if [ "$i" -eq 300 ]; then
            echo "ERROR: Remote Qdrant at $QDRANT_URL not reachable within 300s"
            exit 1
        fi
        sleep 1
    done
fi

case "$MODE" in
    index)
        echo "Running indexing: python3 index_data.py $@"
        exec python3 index_data.py "$@"
        ;;
    serve)
        echo "Starting API and Web UI server on port 8080..."
        exec python3 -m uvicorn api:app --host 0.0.0.0 --port 8080
        ;;
    *)
        echo "Usage: $0 {index|serve} [args...]"
        echo "  index [--pages N] [--workers N] [--rebuild]   Run the indexing pipeline"
        echo "  serve                                          Start the query API"
        exit 1
        ;;
esac
