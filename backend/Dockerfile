# Base image: Python 3.11 slim
FROM python:3.11-slim

# Prevent python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/hf_cache \
    HF_HUB_OFFLINE=1 \
    CHROMA_DIR=/app/Jurixai_db

WORKDIR /app

# Install minimal OS dependencies for Tesseract OCR (English + Hindi) & PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

# Crucial Docker size & memory optimization:
# Install CPU-only PyTorch first (pinned to 2.11.0 to prevent pip from installing bulky CUDA wheels)
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace embedding model at build time into fixed HF_HOME
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create non-root application user
RUN useradd -m -u 1000 appuser

# Copy application files and prebuilt database
COPY backend.py .
COPY Jurixai_db ./Jurixai_db

# Ensure appuser owns and has read+write permissions on Jurixai_db and HF_HOME cache
RUN chown -R appuser:appuser /app/Jurixai_db /app/hf_cache
USER appuser

EXPOSE 8000

# Container liveness health check using Python urllib (no external curl binary needed)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health')" || exit 1

# Launch Uvicorn:
# - Each worker loads sentence-transformers (~350MB) and ChromaDB into memory. Keeping --workers 1 bounds container RAM < 800MB.
# - Proxy headers enabled; FORWARDED_ALLOW_IPS read from env var (defaults to 127.0.0.1, set to platform proxy IP in production).
CMD ["sh", "-c", "exec uvicorn backend:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips \"${FORWARDED_ALLOW_IPS:-127.0.0.1}\""]
