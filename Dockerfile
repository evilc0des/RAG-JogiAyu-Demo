# Stage 1: Build the React UI
FROM node:20-alpine AS ui-builder
WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

# Stage 2: Python Backend
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    QDRANT_URL= \
    OPENAI_BASE_URL=https://inference.do-ai.run/v1 \
    TOKENIZERS_PARALLELISM=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py /app/
COPY ingestion/ /app/ingestion/
COPY --from=ui-builder /app/ui/dist /app/ui/dist

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /app/data

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
