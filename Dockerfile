# ── Base Image ────────────────────────────────────────────────
# Use official Python 3.11 slim image
# 'slim' = smaller than full Python image (no dev tools)
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────
LABEL maintainer="Gurram Mohith Vamsi"
LABEL description="Enterprise Document Intelligence Platform"
LABEL version="1.0"

# ── System Dependencies ───────────────────────────────────────
# Install system packages needed by PyMuPDF and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*
# rm -rf clears apt cache → keeps image small

# ── Working Directory ─────────────────────────────────────────
# All subsequent commands run from /app inside the container
WORKDIR /app

# ── Install Python Dependencies ───────────────────────────────
# Copy requirements FIRST (before code) — Docker layer caching:
# If requirements.txt hasn't changed, Docker skips this layer
# This makes rebuilds much faster
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Application Code ─────────────────────────────────────
# Copy everything else after dependencies
# (so code changes don't invalidate the pip cache layer)
COPY . .

# ── Create Required Directories ───────────────────────────────
RUN mkdir -p uploads chroma_db data

# ── Environment Variables ─────────────────────────────────────
# These are defaults — override with docker-compose.yml
ENV PYTHONUNBUFFERED=1
# PYTHONUNBUFFERED=1 means Python prints logs immediately
# (not buffered) — essential for seeing logs in Docker

ENV PYTHONDONTWRITEBYTECODE=1
# Don't write .pyc files inside container

# ── Expose Port ───────────────────────────────────────────────
# Streamlit runs on 8501 by default
EXPOSE 8501

# ── Health Check ─────────────────────────────────────────────
# Docker checks if the app is alive every 30 seconds
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health \
    || exit 1

# ── Start Command ─────────────────────────────────────────────
# This runs when the container starts
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
# server.address=0.0.0.0 → accept connections from outside container
# server.headless=true   → don't try to open a browser