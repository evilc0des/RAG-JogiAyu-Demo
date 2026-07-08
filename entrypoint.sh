#!/bin/bash
set -e

mkdir -p /app/data

echo "=== Jogi Ayu RAG — Starting API server ==="
echo "OPENAI_BASE_URL=${OPENAI_BASE_URL:-not set}"
echo "QDRANT_URL=${QDRANT_URL:-local disk mode}"
echo ""

exec python -m uvicorn api:app --host 0.0.0.0 --port 8000
