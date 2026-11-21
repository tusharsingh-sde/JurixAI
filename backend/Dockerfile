# Base image: Python 3.11 slim
FROM python:3.11-slim

# Prevent python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/hf_cache \
    CHROMA_DIR=/app/Jurixai_db

WORKDIR /app

# Install minimal OS dependencies for Tesseract OCR (English + Hindi) & PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to avoid bulky CUDA wheels
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and bake embedding model at build time (no dependency on gitignored local_model_cache)
RUN mkdir -p /app/hf_cache /app/local_model_cache && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('/app/local_model_cache/all-MiniLM-L6-v2')"

# Enforce strict offline mode at runtime after model is baked into image
ENV HF_HUB_OFFLINE=1

# Create non-root application user
RUN useradd -m -u 1000 appuser

# Copy application files and prebuilt vector database
COPY backend.py .
COPY Jurixai_db ./Jurixai_db

# Ensure appuser owns ChromaDB, baked model cache, and HF_HOME
RUN chown -R appuser:appuser /app/Jurixai_db /app/local_model_cache /app/hf_cache
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health')" || exit 1

CMD ["sh", "-c", "exec uvicorn backend:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips \"${FORWARDED_ALLOW_IPS:-127.0.0.1}\""]
