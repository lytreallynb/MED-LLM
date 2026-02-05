# MED-LLM Docker Image
# Multi-stage build for smaller final image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY medllm/ ./medllm/
COPY llm.py .
COPY Makefile .

# Create data directories
RUN mkdir -p data/meta data/clean data/eval

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MEDLLM_INDEX_PATH=/app/data/clean/fda.index
ENV MEDLLM_META_PATH=/app/data/clean/fda_meta.jsonl

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run the API server
CMD ["uvicorn", "medllm.server:app", "--host", "0.0.0.0", "--port", "8000"]
