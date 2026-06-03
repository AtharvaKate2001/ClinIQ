# ─────────────────────────────────────────────────────────────────────────────
# ClinIQ — Backend Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY core/ ./core/
COPY agents/ ./agents/
COPY tools/ ./tools/
COPY api/ ./api/
COPY data/samples/ ./data/samples/

# Create persistent data dirs
RUN mkdir -p /app/data/chroma_db /app/data/checkpoints

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
