#!/usr/bin/env bash
# =============================================================================
# Lilly PDF Extractor Agent — One-Command Setup & Launch
# Validates env → Deploys Cortex agents → Generates registry → Starts FastAPI
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

header()  { echo -e "\n${CYAN}═══════════════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}═══════════════════════════════════════════════${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warn()    { echo -e "${YELLOW}⚠️  $1${NC}"; }
fail()    { echo -e "${RED}❌ $1${NC}"; exit 1; }

# =============================================================================
# 0. Load .env if present
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
    set +u  # Disable unbound-var check — .env values may contain $ characters
    # Read .env line-by-line, stripping Windows \r and skipping comments/blanks
    while IFS= read -r line; do
        line=$(echo "$line" | tr -d '\r')
        # skip blanks and comments
        [[ -z "$line" || "$line" == \#* ]] && continue
        # split on the FIRST '=' only (values may contain '=')
        key="${line%%=*}"
        value="${line#*=}"
        key=$(echo "$key" | xargs)                     # trim whitespace
        value=$(echo "$value" | sed 's/^ *//;s/ *$//')  # trim whitespace
        # strip surrounding quotes from value
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        [[ -z "$key" ]] && continue
        export "$key=$value"
    done < .env
    set -u
    success "Loaded .env"
else
    warn ".env file not found — relying on shell environment variables"
fi

# =============================================================================
# 1. Validate Required Environment Variables
# =============================================================================
header "Step 1/4 — Checking Required Environment Variables"

REQUIRED_VARS=(
    "OWNER_EMAIL"
    "CORTEX_COOKIE"
)

MISSING=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING+=("$var")
    else
        success "$var is set"
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    fail "Missing required environment variables:\n  ${MISSING[*]}\n\nAdd them to your .env file and retry."
fi

# # Optional vars — just inform
# OPTIONAL_VARS=("CORTEX_BASE_URL" "AWS_S3_BUCKET_NAME" "GCP_BUCKET_NAME")
# echo ""
# for var in "${OPTIONAL_VARS[@]}"; do
#     if [ -z "${!var:-}" ]; then
#         warn "$var not set (optional — using default/disabled)"
#     else
#         success "$var is set"
#     fi
# done

# =============================================================================
# 2. Check Python & dependencies
# =============================================================================
header "Step 2/4 — Checking Python Environment"

# On Windows, check the .venv first (python3 doesn't exist, and bare 'python'
# may be the Microsoft Store alias which is not a real interpreter).
if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON=$(command -v python3)
elif command -v python &>/dev/null; then
    PYTHON=$(command -v python)
else
    fail "Python is not installed or not on PATH. Create a venv first:\n  python -m venv .venv && source .venv/Scripts/activate"
fi

success "Python found: $($PYTHON --version 2>&1)"

# Quick check that key packages are installed
$PYTHON -c "import fastapi, langgraph, httpx, msal" 2>/dev/null || {
    warn "Some dependencies are missing. Installing from requirements.txt..."
    $PYTHON -m pip install -r requirements.txt
}
success "Dependencies OK"

# =============================================================================
# 3. Deploy Cortex Agents & Generate Registry
# =============================================================================
header "Step 3/4 — Deploying Cortex Agents"

echo "Deploying agents for owner: $OWNER_EMAIL"
$PYTHON manage_cortex_agent/manage_agents.py deploy
success "Agent deployment complete"

echo ""
echo "Generating agent_registry.py..."
$PYTHON manage_cortex_agent/manage_agents.py registry
success "Registry generated"

# =============================================================================
# 4. Start FastAPI Server (background) + cleanup on exit
# =============================================================================
header "Step 4/4 — Starting FastAPI Server"

UVICORN_PID=""

cleanup() {
    echo ""
    header "Shutting Down"

    # Stop uvicorn
    if [ -n "$UVICORN_PID" ] && kill -0 "$UVICORN_PID" 2>/dev/null; then
        echo "Stopping FastAPI server (PID $UVICORN_PID)..."
        kill "$UVICORN_PID" 2>/dev/null
        wait "$UVICORN_PID" 2>/dev/null || true
        success "Server stopped"
    fi

    # Delete all deployed agents
    echo ""
    echo "Cleaning up Cortex agents..."
    $PYTHON manage_cortex_agent/manage_agents.py delete <<< "y"
    success "All agents deleted. Goodbye!"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Launching on http://0.0.0.0:8000"
echo "API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server and delete all deployed agents."
echo ""

$PYTHON -m uvicorn app:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!
success "Server started (PID $UVICORN_PID)"

# Wait for the background process — trap will fire on Ctrl+C
wait "$UVICORN_PID" 2>/dev/null || true

# If uvicorn exits on its own (crash/etc.), still clean up
cleanup
