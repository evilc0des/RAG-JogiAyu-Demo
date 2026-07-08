#!/bin/bash
set -e

mkdir -p /app/data

MODE="${1:-serve}"
shift || true

case "$MODE" in
    serve)
        echo "=== Jogi Ayu RAG — Starting API server ==="
        echo "OPENAI_BASE_URL=${OPENAI_BASE_URL:-not set}"
        echo "QDRANT_URL=${QDRANT_URL:-local disk mode}"
        echo ""
        exec python -m uvicorn api:app --host 0.0.0.0 --port 8000
        ;;
    benchmark)
        echo "=== Jogi Ayu RAG — Running benchmarks ==="
        exec python benchmark.py "$@"
        ;;
    ingest)
        echo "=== Jogi Ayu RAG — Running ingestion ==="
        exec python cli_ingestion.py "$@"
        ;;
    shell)
        exec /bin/bash
        ;;
    *)
        echo "Usage: $0 {serve|benchmark|ingest|shell} [args...]"
        echo "  serve                           Start the API server (default)"
        echo "  benchmark [--layers all] ...    Run benchmark suite"
        echo "  ingest [--file PATH] ...        Run data ingestion"
        echo "  shell                           Open a bash shell"
        exit 1
        ;;
esac
