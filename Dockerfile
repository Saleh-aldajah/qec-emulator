# QEC Emulator v2.0.0 — Docker image
# Provides: CLI, REST API server, and all simulation runners
#
# Usage:
#   docker build -t qec-emulator .
#   docker run -p 8765:8765 qec-emulator              # API server
#   docker run qec-emulator qec-emulator sweep --code BB72 --trials 1000
#   docker run qec-emulator qec-emulator verify

FROM python:3.11-slim

LABEL org.opencontainers.image.title="QEC Emulator"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.description="BB code qLDPC decoder benchmarking emulator"
LABEL org.opencontainers.image.authors="Dr. Saleh H. AlDaajeh <S.aldaajeh@gmail.com>"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies first (better layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    numpy>=1.24 \
    ldpc>=2.0 \
    pymatching>=2.0 \
    stim>=1.12 \
    networkx>=3 \
    matplotlib>=3.7 \
    fastapi>=0.110 \
    uvicorn>=0.27 \
    typer>=0.12 \
    rich>=13 \
    pyyaml>=6 \
    httpx>=0.24

# Copy source
COPY . .
RUN pip install --no-cache-dir -e .

# Expose API port
EXPOSE 8765

# Default: run the API server
CMD ["qec-emulator", "server", "--host", "0.0.0.0", "--port", "8765"]
