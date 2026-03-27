# ============================================================
# lilly-pdf-extractor-agent
# Python 3.12.4 | FastAPI + uvicorn | host 0.0.0.0:8000
# ============================================================

# python:3.12.4 (full, not slim) is used intentionally:
# - slim strips gcc, libffi-dev, and other build tools
# - slim requires apt-get to reinstall them, which needs deb.debian.org
# - in corporate networks deb.debian.org is often unreachable
# - the full image (buildpack-deps:bookworm) ships with gcc, libffi, make
#   already present, so no apt-get step is needed at all
# - PyMuPDF 1.26.7+ ships a self-contained wheel (bundled MuPDF), no libgl1 needed
FROM python:3.12.4

# Prevent .pyc files; ensure stdout/stderr are unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ------------------------------------------------------------------
# pip configuration — must come BEFORE pip install.
# pip.conf carries the extra-index-url pointing to the internal
# Lilly Artifactory registry where langchain-cortex lives.
# ⚠  pip.conf embeds credentials — do NOT push to a public registry
# ------------------------------------------------------------------
COPY pip.conf /etc/pip.conf

# ------------------------------------------------------------------
# Install Python dependencies as a dedicated layer.
# Re-runs ONLY when requirements.txt changes — fast rebuilds.
# ------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------
# Post-install verification — fails the build if any critical
# package is missing or its import is broken.
# ------------------------------------------------------------------
RUN python -c "\
import langchain_cortex; \
import langchain; \
import langgraph; \
import fastapi; \
import fitz; \
import uvicorn; \
import boto3; \
import google.cloud.storage; \
print('✅ All critical packages verified successfully')"

# ------------------------------------------------------------------
# Copy application source
# ------------------------------------------------------------------
COPY app.py main.py batch_submit.py json2md.py ./
COPY agents/              agents/
COPY core/                core/
COPY utils/               utils/
COPY prompts/             prompts/

# ------------------------------------------------------------------
# Environment & credentials
# .env  → loaded by python-dotenv at runtime (load_dotenv())
# gcp-service-account-key.json → used by CloudStorageManager
#         when GOOGLE_APPLICATION_CREDENTIALS points to this path
# ⚠  This image contains secrets — do NOT push to a public registry
# ------------------------------------------------------------------
COPY .env                          .env
COPY gcp-service-account-key.json  gcp-service-account-key.json

# Pre-create runtime write directories so the app never hits a
# missing-directory error on first run
RUN mkdir -p logs output_json memory/pdf_cache

# Declare persistent storage mount points.
# Use -v flags at runtime to mount these to the host (or named volumes)
# so that SQLite DBs, output JSON, and logs survive container restarts.
VOLUME ["/app/memory", "/app/output_json", "/app/logs"]

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
