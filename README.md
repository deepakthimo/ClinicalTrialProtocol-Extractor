# Lilly PDF Extractor Agent

An asynchronous, multi-agent PDF extraction pipeline built with **LangGraph** and **Cortex LLMs**, exposed via **FastAPI**. Designed to extract clinical trial protocol data from PDF documents and produce structured JSON training datasets with high-fidelity markdown output.

---

## Key Features

- **Multi-Agent LangGraph Orchestration** — 3-tier graph hierarchy (Master → Crop → Page) with 10+ specialized LLM agents
- **Fully Asynchronous** — `httpx.AsyncClient` + `asyncio` for non-blocking I/O across all agent nodes
- **Resilient Job Persistence** — SQLite-backed job tracking with auto-resume on server restart via LangGraph checkpointing
- **Adaptive Retry Logic** — Self-correcting page extraction loop that diagnoses error type (layout vs. rule) and routes to the appropriate retry node
- **Circuit Breakers** — PDF quality and synopsis sufficiency gates that halt processing early on unsuitable documents
- **Dynamic Crop Memory** — Learns and remembers optimal bounding box coordinates per clinical sponsor
- **Sponsor-Aware Prompts** — Loads sponsor-specific extraction prompts at runtime with fallback to general templates
- **Multi-Cloud Storage** — Unified `CloudStorageManager` supporting AWS S3, Google Cloud Storage, and local fallback
- **Structured Output Enforcement** — Pydantic schema injection into LLM prompts for guaranteed JSON compliance
- **Round-Robin Load Balancing** — Thread-safe distribution of LLM calls across multiple Cortex agent copies

---

## Architecture Overview

### High-Level System Flow

![High level view](utils\readme_images\high-level-system-flow.png)

### Master Graph — Orchestrator (10 Nodes)

The master graph manages the end-to-end extraction pipeline. It processes the PDF through sequential phases with two circuit breaker gates that can halt processing early.

![Master Graph](utils\readme_images\master_agent.png)

### Page Agent — Per-Page Extraction (4 Nodes + Retry Loop)

Each page goes through a VLM extraction → LLM reconciliation → VLM validation cycle. On failure, the error type determines whether to retry extraction (layout issues) or reconciliation (rule-following issues). After exhausting retries, a best-of-N judge selects the best attempt.

![Page Agent](utils\readme_images\crop_agent.png)

### Crop Agent — Bounding Box Calculation (3 Nodes + Retry Loop)

The crop agent determines the optimal bounding box to exclude headers/footers from PDF pages. It leverages sponsor-specific memory to avoid recalculating known-good crop boxes.

![Crop Agent](utils\readme_images\crop_agent.png)

---

## Project Structure

```
lilly-pdf-extractor-agent/
├── app.py                          # FastAPI server, endpoints, background task manager, auto-resume
├── main.py                         # CLI entry point for standalone testing
├── requirements.txt                # Python dependencies
├── json2md.py                      # Utility: convert output JSON to markdown
│
├── agents/
│   ├── master_graph.py             # LangGraph orchestrator (10 nodes)
│   ├── page_agent.py               # Per-page extraction graph (4 nodes + retry)
│   ├── crop_agent.py               # Bounding box calculation graph (3 nodes + retry)
│   ├── agent_registry.py           # Maps agent roles → Cortex model copies (load balancing)
│   └── cortex_llm_config.py        # Cortex API wrapper (auth, retry, structured output)
│
├── core/
│   ├── config.py                   # Environment variables and tuning knobs
│   ├── state.py                    # TypedDict state schemas + Pydantic validation models
│   ├── initial_state.py            # Factory: creates base state dict for pipeline
│   ├── job_manager.py              # SQLite job persistence (PENDING → IN_PROGRESS → COMPLETED → FAILED)
│   └── logger.py                   # Dual-output logger (console + file)
│
├── prompts/
│   ├── synopsis_sys_inst.txt       # System prompt for synopsis extraction
│   └── sponsors/
│       ├── section/                # Sponsor-specific section extraction prompts
│       │   ├── general.txt
│       │   └── pfizer.txt
│       └── synopsis/               # Sponsor-specific synopsis extraction prompts
│           ├── general.txt
│           └── Boehringer Ingelheim.txt
│
├── utils/
│   ├── pdf_parser.py               # PDF download, TOC extraction, page merging, crop utilities
│   ├── cloud_storage.py            # Unified AWS S3 / GCP / local storage manager
│   └── title_card.py               # ASCII art banner display
│
├── manage_cortex_agent/
│   ├── manage_agents.py            # Script to provision/manage Cortex agent copies
│   ├── agents_config.json          # Agent deployment configuration
│   └── README.md                   # Agent management documentation
│
├── memory/                         # Runtime data (gitignored)
│   ├── sponsor_based_crop_memory.json  # Learned crop boxes per sponsor
│   ├── job_states.db               # SQLite job tracking database
│   ├── master_graph_checkpoints.db # LangGraph checkpoint database
│   └── pdf_cache/                  # Downloaded PDF cache
│
└── tests/
    ├── conftest.py                 # Pytest fixtures
    ├── test_concurrency.py         # Concurrency safety tests
    ├── test_graceful_degradation.py # Resilience tests
    ├── test_load_api.py            # API load tests
    └── locustfile.py               # Locust load testing config
```

---

## Getting Started

The setup script handles the full lifecycle — from environment validation to agent cleanup — in a single command. This is the recommended way for team members to run the pipeline.

### Step 1: Clone & Install

```bash
git clone <repository-url>
cd lilly-pdf-extractor-agent

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

Create a `.env` file in the project root with your credentials:

```env
# ── Your Identity (Required) ──
OWNER_EMAIL=your.name@network.lilly.com

# ── Cortex Agent Management (Required) ──
CORTEX_COOKIE=your-cortex-cookie

# ── Cloud Storage (Optional — falls back to local) ──
AWS_S3_BUCKET_NAME=your-s3-bucket
GCP_BUCKET_NAME=your-gcp-bucket

# ── Tuning (Optional — sensible defaults provided) ──
MAX_CONCURRENT_PDFS=1
MAX_CONCURRENT_SECTIONS=5
LLM_TIMEOUT_SECONDS=180
LLM_TIMEOUT_MULTIMODAL_SECONDS=240
LLM_RETRY_ATTEMPTS=4
PAGE_RETRY_ATTEMPTS=2
PAGE_RETRY_BACKOFF_SECONDS=60
PAGE_THROTTLE_SECONDS=3
DEBUG=false
```

<details>
<summary><b>Environment Variables Reference (click to expand)</b></summary>

| Variable | Required | Default | Description |
|:---|:---:|:---:|:---|
| `OWNER_EMAIL` | Yes | — | Your Lilly email — used to prefix agent names and set ownership |
| `CORTEX_COOKIE` | Yes | — | Authentication cookie for the Cortex agent management API |
| `CORTEX_CLIENT_ID` | Yes | — | Azure AD application client ID |
| `CORTEX_CLIENT_SECRET` | Yes | — | Azure AD client secret |
| `CORTEX_AUTHORITY` | Yes | — | Azure AD authority URL |
| `CORTEX_SCOPE` | Yes | — | Cortex API scope for token acquisition |
| `CORTEX_BASE_URL` | No | `https://gateway.apim...` | Cortex APIM gateway base URL |
| `AWS_S3_BUCKET_NAME` | No | — | AWS S3 bucket for result uploads |
| `GCP_BUCKET_NAME` | No | — | GCP bucket for result uploads |
| `MAX_CONCURRENT_PDFS` | No | `1` | Max PDFs processed simultaneously |
| `MAX_CONCURRENT_SECTIONS` | No | `5` | Max sections processed in parallel per PDF |
| `LLM_TIMEOUT_SECONDS` | No | `180` | Timeout for text-only LLM calls (seconds) |
| `LLM_TIMEOUT_MULTIMODAL_SECONDS` | No | `240` | Timeout for multimodal LLM calls (seconds) |
| `LLM_RETRY_ATTEMPTS` | No | `4` | Max retry attempts for LLM API calls |
| `PAGE_RETRY_ATTEMPTS` | No | `2` | Outer retry attempts per page agent invocation |
| `PAGE_RETRY_BACKOFF_SECONDS` | No | `60` | Backoff delay between page retries |
| `PAGE_THROTTLE_SECONDS` | No | `3` | Delay between sequential page processing |
| `DEBUG` | No | `false` | Enable debug mode (crop visualization, verbose logs) |

</details>

> **Note:** Each team member uses their own `OWNER_EMAIL`. This prefixes all deployed agent names with your identity (e.g. `deepaktm-pageagent-extractor`), keeping each person's agents isolated.

### Step 3: Run the Setup Script

**Windows:**

```cmd
setup.bat
```

**Linux / macOS:**

```bash
chmod +x setup.sh
./setup.sh
```

### What the Script Does

```
setup.sh / setup.bat
│
├── Step 1: Validate environment variables (OWNER_EMAIL, CORTEX_COOKIE, ...)
├── Step 2: Verify Python installation & install missing dependencies
├── Step 3: Deploy Cortex agents (prefixed with your name)
│           └── Generate agents/agent_registry.py
├── Step 4: Start FastAPI server in the background
│           └── http://0.0.0.0:8000 (API docs at /docs)
│
│   ... server is running, send extraction requests ...
│
├── Ctrl+C pressed
│   ├── Stop FastAPI server
│   └── Delete ALL deployed Cortex agents (clean teardown)
└── Exit
```

The full lifecycle is managed automatically — when you're done, just press **Ctrl+C** and all your Cortex agents are cleaned up.

### Step 4: Submit an Extraction Request

With the server running, submit a PDF via curl or any HTTP client:

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"pdf_url": "https://example.com/clinical-protocol.pdf", "sponsor_name": "Eli Lilly"}'
```

Then check the status:

```bash
# Replace <job_id> with the ID from the response
curl http://localhost:8000/api/v1/status/<job_id>
```

And retrieve results when complete:

```bash
curl http://localhost:8000/api/v1/results/<job_id>
```

### Manual Start (Server Only)

If agents are already deployed and the registry is generated, you can start just the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

> **Warning:** Manual start does not set up the Ctrl+C cleanup trap. You must manually delete agents with `python manage_cortex_agent/manage_agents.py delete` when done.

On startup, the server automatically checks for interrupted jobs and resumes them.

---

## API Reference

### Submit Single Extraction

```
POST /api/v1/extract
```

**Request Body:**

```json
{
  "pdf_url": "https://example.com/path/to/clinical-protocol.pdf",
  "sponsor_name": "Eli Lilly"
}
```

**Response** `202 Accepted`:

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Extraction job successfully queued.",
  "status_url": "/api/v1/status/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

---

### Submit Batch Extraction

```
POST /api/v1/extract/batch
```

**Request Body:**

```json
{
  "requests": [
    { "pdf_url": "https://example.com/protocol-1.pdf", "sponsor_name": "Pfizer" },
    { "pdf_url": "https://example.com/protocol-2.pdf", "sponsor_name": "Boehringer Ingelheim" }
  ]
}
```

**Response** `202 Accepted`:

```json
{
  "batch_id": "b1c2d3e4-f5a6-7890-bcde-fa1234567890",
  "jobs": [
    {
      "job_id": "...",
      "message": "Extraction job successfully queued.",
      "status_url": "/api/v1/status/..."
    },
    {
      "job_id": "...",
      "message": "Extraction job successfully queued.",
      "status_url": "/api/v1/status/..."
    }
  ]
}
```

---

### Check Job Status

```
GET /api/v1/status/{job_id}
```

**Response** `200 OK`:

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "error_message": null,
  "result_url": "/api/v1/results/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

Possible `status` values: `pending`, `in_progress`, `completed`, `failed`

---

### Retrieve Results

```
GET /api/v1/results/{job_id}
```

**Response** `200 OK`: Returns the final JSON dataset (see [Output Format](#output-format) below).

---

### View Live Logs

```
GET /api/v1/logs?lines=200
```

Returns the last N lines of the extraction log as plain text. Default: 200 lines.

---

### Health Check

```
GET /health
```

**Response** `200 OK`:

```json
{
  "status": "ok",
  "timestamp": "2026-03-23T12:00:00+00:00"
}
```

---

## Monitoring Logs

To watch real-time extraction progress in a separate terminal:

**Windows (PowerShell):**

```powershell
Get-Content -Path logs\agent_run.log -Wait -Tail 20
```

**Linux / macOS:**

```bash
tail -f logs/agent_run.log
```

---

## Output Format

The pipeline produces a JSON object where each key is a normalized section title, and the value contains an `instruction` / `input` / `output` triplet for training data:

```json
{
  "study design": {
    "instruction": "You are an expert Clinical Medical Writer. Below is the Synopsis for a clinical trial. Based strictly on this information, generate the content for the section: Study Design.",
    "input": "Protocol Number: ABC-1234\nStudy Title: A Phase 3 Randomized...\n...",
    "output": "# Study Design\n\nThis is a Phase 3, randomized, double-blind..."
  },
  "study population": {
    "instruction": "You are an expert Clinical Medical Writer. Below is the Synopsis for a clinical trial. Based strictly on this information, generate the content for the section: Study Population.",
    "input": "Protocol Number: ABC-1234\nStudy Title: A Phase 3 Randomized...\n...",
    "output": "# Study Population\n\nApproximately 500 adult patients aged 18-75..."
  }
}
```

**12 Required Sections**: study design, study population, objectives, endpoints, treatment, procedures, statistical analysis, efficacy, safety, pharmacokinetics, biomarkers, data monitoring

---

## Data Persistence

| Store | Location | Purpose |
|:---|:---|:---|
| Job Tracking | `memory/job_states.db` | SQLite database for job status persistence across restarts |
| Graph Checkpoints | `memory/master_graph_checkpoints.db` | LangGraph mid-pipeline checkpoints for resumption |
| Crop Memory | `memory/sponsor_based_crop_memory.json` | Learned bounding box coordinates per sponsor (max 3 per sponsor) |
| PDF Cache | `memory/pdf_cache/` | Downloaded PDFs cached by job ID |
| Final Output | `output_json/` | Generated JSON datasets by NCT number + timestamp |
