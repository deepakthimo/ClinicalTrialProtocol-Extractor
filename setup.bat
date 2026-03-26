@echo off
REM =============================================================================
REM Lilly PDF Extractor Agent — One-Command Setup & Launch (Windows)
REM Validates env → Deploys Cortex agents → Generates registry → Starts FastAPI
REM =============================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"

REM =============================================================================
REM 0. Load .env (simple key=value parser)
REM =============================================================================
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        REM Skip comments and blank lines
        set "line=%%A"
        if not "!line:~0,1!"=="#" (
            if not "%%A"=="" set "%%A=%%B"
        )
    )
    echo [OK] Loaded .env
) else (
    echo [WARN] .env file not found — relying on system environment variables
)

REM =============================================================================
REM 1. Validate Required Environment Variables
REM =============================================================================
echo.
echo ===============================================
echo   Step 1/4 — Checking Required Environment Variables
echo ===============================================

set "MISSING="
set "HAS_ERROR=0"

for %%V in (OWNER_EMAIL CORTEX_COOKIE CORTEX_CLIENT_ID CORTEX_CLIENT_SECRET CORTEX_AUTHORITY CORTEX_SCOPE) do (
    if "!%%V!"=="" (
        echo [FAIL] %%V is NOT set
        set "MISSING=!MISSING! %%V"
        set "HAS_ERROR=1"
    ) else (
        echo [OK]   %%V is set
    )
)

if "!HAS_ERROR!"=="1" (
    echo.
    echo ERROR: Missing required environment variables:!MISSING!
    echo Add them to your .env file and retry.
    exit /b 1
)

REM Optional vars
echo.
for %%V in (CORTEX_BASE_URL AWS_S3_BUCKET_NAME GCP_BUCKET_NAME) do (
    if "!%%V!"=="" (
        echo [WARN] %%V not set ^(optional^)
    ) else (
        echo [OK]   %%V is set
    )
)

REM =============================================================================
REM 2. Check Python
REM =============================================================================
echo.
echo ===============================================
echo   Step 2/4 — Checking Python Environment
echo ===============================================

where python >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Python is not installed or not on PATH
    exit /b 1
)

python --version
echo [OK] Python found

python -c "import fastapi, langgraph, httpx, msal" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Some dependencies missing. Installing...
    pip install -r requirements.txt
)
echo [OK] Dependencies OK

REM =============================================================================
REM 3. Deploy Cortex Agents & Generate Registry
REM =============================================================================
echo.
echo ===============================================
echo   Step 3/4 — Deploying Cortex Agents
echo ===============================================

echo Deploying agents for owner: %OWNER_EMAIL%
python manage_cortex_agent\manage_agents.py deploy
if errorlevel 1 (
    echo [FAIL] Agent deployment failed
    exit /b 1
)
echo [OK] Agent deployment complete

echo.
echo Generating agent_registry.py...
python manage_cortex_agent\manage_agents.py registry
if errorlevel 1 (
    echo [FAIL] Registry generation failed
    exit /b 1
)
echo [OK] Registry generated

REM =============================================================================
REM 4. Start FastAPI Server
REM =============================================================================
echo.
echo ===============================================
echo   Step 4/4 — Starting FastAPI Server
echo ===============================================

echo Launching on http://0.0.0.0:8000
echo API docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server and delete all deployed agents.
echo.

start /b python -m uvicorn app:app --host 0.0.0.0 --port 8000
echo [OK] Server started in background
echo.
echo Waiting for Ctrl+C to shut down...
echo.

REM Keep the script alive — pause responds to any key, so we loop
:waitloop
timeout /t 3600 /nobreak >nul 2>&1
goto waitloop

:cleanup
echo.
echo ===============================================
echo   Shutting Down
echo ===============================================

REM Kill uvicorn
echo Stopping FastAPI server...
taskkill /f /im python.exe /fi "WINDOWTITLE eq uvicorn*" >nul 2>&1
for /f "tokens=5" %%P in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%P >nul 2>&1
)
echo [OK] Server stopped

REM Delete agents
echo.
echo Cleaning up Cortex agents...
echo y | python manage_cortex_agent\manage_agents.py delete
echo [OK] All agents deleted. Goodbye!
exit /b 0
