# Stage 1: Build the UI
FROM node:20 AS ui-builder
WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

# Stage 2: Python Backend
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    QDRANT_URL=http://localhost:6333 \
    HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    ca-certificates \
    zlib1g \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L -o /tmp/qdrant.tar.gz \
        https://github.com/qdrant/qdrant/releases/download/v1.9.7/qdrant-x86_64-unknown-linux-gnu.tar.gz \
    && tar xzf /tmp/qdrant.tar.gz -C /usr/local/bin \
    && rm /tmp/qdrant.tar.gz

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 \
        -r requirements.txt && \
    pip install --force-reinstall --no-deps onnxruntime-gpu==1.21.0

# Pre-download the embedding model at build time so startup is instant
# Use huggingface_hub directly to avoid loading onnxruntime-gpu (no GPU during build)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-small-en-v1.5')"

COPY *.py /app/
COPY scripts/ /app/scripts/
COPY --from=ui-builder /app/ui/dist /app/ui/dist

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
