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
    LLM_PORT=8001 \
    LLM_MODEL_REPO="Qwen/Qwen2.5-7B-Instruct-GGUF" \
    LLM_MODEL_FILE="qwen2.5-7b-instruct-q4_k_m.gguf" \
    LLM_N_GPU_LAYERS=-1 \
    LLM_MAX_TOKENS=8192 \
    LLM_THREADS=8 \
    OPENAI_BASE_URL=http://localhost:8001/v1 \
    OPENAI_API_KEY=not-needed \
    HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    ca-certificates \
    zlib1g \
    libcudnn9-cuda-12 \
    rsync \
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

# Install llama-cpp-python with CUDA acceleration (OpenAI-compatible API server on port 8001)
# Uses pre-built CUDA 12.4 wheels — no compilation needed, ~200 MB total install
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Pre-download the embedding model at build time so startup is instant
# Use huggingface_hub directly to avoid loading onnxruntime-gpu (no GPU during build)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-small-en-v1.5')"

# Pre-download the LLM GGUF model at build time so first startup is instant
# Skip by default to keep Docker image small (~4.7 GB saved); model downloads at first startup
ARG PREDOWNLOAD_LLM=false
RUN if [ "$PREDOWNLOAD_LLM" = "true" ]; then \
        python3 -c "\
from huggingface_hub import hf_hub_download; \
repo = '$LLM_MODEL_REPO'; \
file = '$LLM_MODEL_FILE'; \
print(f'Downloading {repo}/{file}...'); \
hf_hub_download(repo, file)"; \
    fi

COPY *.py /app/
COPY scripts/ /app/scripts/
COPY --from=ui-builder /app/ui/dist /app/ui/dist

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080 8001

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
