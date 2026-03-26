import os
import uuid
import json
import re  # Added for NCT extraction
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
import traceback
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import BackgroundTasks

from fastapi.responses import PlainTextResponse

from core.job_manager import job_manager, JobRecord, JobStatus

# 1. FIXED: Import workflow and AsyncSqliteSaver exactly like main.py
from agents.master_graph import workflow
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from core.initial_state import create_initial_state

from core.logger import setup_logger
from core.config import MAX_CONCURRENT_PDFS
logger = setup_logger()

logger.info("🔥 APP.PY MODULE LOADED")

# PDF-level concurrency gate — prevents overloading LLM agents when many jobs arrive at once
_pdf_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PDFS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 App starting...")

    # ✅ STARTUP: Check for stuck/failed jobs and resume them
    logger.info("🔄 Checking for stuck jobs...")
    stuck_jobs = job_manager.get_incomplete_jobs()

    if not stuck_jobs:
        logger.info("✅ No stuck jobs found.")
    else:
        logger.info(f"🔄 Found {len(stuck_jobs)} stuck jobs. Attempting to auto-resume...")
        for job in stuck_jobs:
            logger.debug(f"🔄 Resuming Job Registry Entry: {job.model_dump_json()}")
            logger.info(f"🔄 Resuming Job [{job.job_id}]")

            req = ExtractRequest(
                pdf_url=job.pdf_url,
                sponsor_name=job.sponsor_name
            )

            asyncio.create_task(
                background_extract_task(job.job_id, req)
            )

    yield  # App is now running and serving requests

    # SHUTDOWN
    logger.info("🛑 Server shutting down.")

app = FastAPI(
    title="Lilly PDF Extractor Agent API",
    description="Multi-agent asynchronous PDF clinical protocol extraction using Cortex API.",
    version="1.0.0",
    lifespan=lifespan
)

# ==========================================
# Schema Definitions
# ==========================================
class ExtractRequest(BaseModel):
    pdf_url: str
    sponsor_name: str

class ExtractResponse(BaseModel):
    job_id: str
    message: str
    status_url: str

class BatchExtractRequest(BaseModel):
    requests: List[ExtractRequest]

class BatchExtractResponse(BaseModel):
    batch_id: str
    jobs: List[ExtractResponse]

class StatusResponse(BaseModel):
    job_id: str
    status: str
    error_message: Optional[str] = None
    result_url: Optional[str] = None

# ==========================================
# Background Processing
# ==========================================
async def background_extract_task(job_id: str, req: ExtractRequest):
    """
    Executes the LangGraph Master pipeline in the background.
    Uses _pdf_semaphore to limit how many PDFs run concurrently.
    """
    try:
        job_manager.update_job_status(job_id, JobStatus.IN_PROGRESS)
        
        # Extract NCT number for logging and file naming
        match = re.search(r"NCT\d+", req.pdf_url)
        nct_number = match.group() if match else f"JOB_{job_id[:8]}"

        # 1. Prepare Initial State Space
        initial_state = create_initial_state(
            job_id=job_id,
            pdf_url=req.pdf_url,
            sponsor_name=req.sponsor_name
        )

        logger.debug(f"🚀 Job [{job_id}]: State initialization complete. Final Payload Keys: {list(initial_state.keys())}")
        logger.info(f"🚀 Job [{job_id}]: Starting LangGraph pipeline for {req.pdf_url}...")
        
        # 2. Acquire PDF-level concurrency slot before running the pipeline
        async with _pdf_semaphore:
            logger.info(f"🔓 Job [{job_id}]: Acquired PDF slot (limit: {MAX_CONCURRENT_PDFS})")
            async with AsyncSqliteSaver.from_conn_string(r"C:\Users\DeepakTM\Music\Projects\lilly-pdf-extractor-agent\memory\master_graph_checkpoints.db") as memory:
                master_app = workflow.compile(checkpointer=memory)
                
                config = {
                    "configurable": {"thread_id": job_id},
                    "run_name": f"Extraction Pipeline: {nct_number}"
                }
                logger.debug("Executing main graph invocation (ainvoke)...")
                
                result_state = await master_app.ainvoke(initial_state, config=config)
        
        # 3. Handle PDF Quality Failures — upload an abort record to GCP before halting
        if result_state.get("pdf_status") == "FAIL":
            abort_node = result_state.get("abort_node", "unknown")
            abort_reason = result_state.get("abort_reason", "No reason captured")
            failure_payload = {
                "job_id": job_id,
                "pdf_url": req.pdf_url,
                "sponsor_name": req.sponsor_name,
                "status": "ABORTED",
                "abort_node": abort_node,
                "abort_reason": abort_reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            try:
                from utils.cloud_storage import storage_manager
                abort_filename = f"aborted/{nct_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ABORTED.json"
                storage_manager.upload_json(abort_filename, failure_payload)
                logger.info(f"☁️ Job [{job_id}]: Abort status uploaded to GCP -> {abort_filename}")
            except Exception as cloud_e:
                logger.warning(f"⚠️ Job [{job_id}]: Could not upload abort status to GCP: {cloud_e}")
            raise Exception(f"Pipeline aborted at node '{abort_node}': {abort_reason}")

        # 4. Save Final Outputs
        final_json = result_state.get("final_dataset", {})
        
        output_dir = "output_json"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{nct_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(final_json, f, indent=4, ensure_ascii=False)
            
        # (Optional) Upload to Cloud
        try:
            from utils.cloud_storage import storage_manager
            cloud_uri = storage_manager.upload_json(f"output_json/{filename}", final_json)
            logger.info(f"✅ Job [{job_id}]: Cloud Upload Success -> {cloud_uri}")
        except Exception as cloud_e:
            logger.warning(f"⚠️ Job [{job_id}]: Cloud upload failed/disabled. Local only. Error: {cloud_e}")
            
        logger.info(f"✅ Job [{job_id}]: Complete. Saved locally to {filepath}")
        job_manager.update_job_status(job_id, JobStatus.COMPLETED, result_path=filepath)

    except Exception as e:
        logger.error(f"❌ Job [{job_id}]: Failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        job_manager.update_job_status(job_id, JobStatus.FAILED, error=str(e))

# ==========================================
# API Endpoints
# ==========================================

# Maintains strong references to background tasks to prevent garbage collection
active_tasks = set()

def _register_job(req: ExtractRequest) -> str:
    """Register a new job in SQLite and return the job_id."""
    job_id = str(uuid.uuid4())
    job_record = JobRecord(
        job_id=job_id,
        status=JobStatus.PENDING,
        pdf_url=req.pdf_url,
        sponsor_name=req.sponsor_name,
        created_at=datetime.now(timezone.utc).isoformat()
    )
    job_manager.create_job(job_record)
    logger.debug(f"API: Job {job_id} registered in database. Source URL: {req.pdf_url}")
    return job_id

def _launch_job_task(job_id: str, req: ExtractRequest):
    """Fire a background asyncio task for the job (keeps a strong ref to prevent GC)."""
    task = asyncio.create_task(background_extract_task(job_id, req))
    active_tasks.add(task)
    task.add_done_callback(active_tasks.discard)

async def initiate_job(req: ExtractRequest, background_tasks: BackgroundTasks) -> ExtractResponse:
    job_id = _register_job(req)
    
    # Single-job endpoint: let FastAPI handle the background thread safely
    background_tasks.add_task(background_extract_task, job_id, req)
    
    return ExtractResponse(
        job_id=job_id,
        message="Extraction job successfully queued.",
        status_url=f"/api/v1/status/{job_id}"
    )

@app.post("/api/v1/extract", response_model=ExtractResponse, status_code=202)
async def submit_extraction_job(req: ExtractRequest, background_tasks: BackgroundTasks):
    return await initiate_job(req, background_tasks)

@app.post("/api/v1/extract/batch", response_model=BatchExtractResponse, status_code=202)
async def submit_batch_extraction(batch_req: BatchExtractRequest):
    batch_id = str(uuid.uuid4())
    jobs = []
    
    for req in batch_req.requests:
        # Register each job in DB
        job_id = _register_job(req)
        
        # Fire as a concurrent asyncio task (semaphore gates actual execution)
        _launch_job_task(job_id, req)
        
        jobs.append(ExtractResponse(
            job_id=job_id,
            message="Extraction job successfully queued.",
            status_url=f"/api/v1/status/{job_id}"
        ))
    
    logger.info(f"Batch [{batch_id}]: Launched {len(jobs)} concurrent jobs (PDF limit: {MAX_CONCURRENT_PDFS})")
    return BatchExtractResponse(
        batch_id=batch_id,
        jobs=jobs
    )

@app.get("/api/v1/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
        
    res = StatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        error_message=job.error_message
    )
    
    if job.status == JobStatus.COMPLETED:
        res.result_url = f"/api/v1/results/{job_id}"
        
    return res

@app.get("/api/v1/results/{job_id}")
async def get_job_results(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job is not completed yet. Current status: {job.status}")
        
    if not job.result_path or not os.path.exists(job.result_path):
        raise HTTPException(status_code=404, detail="Result file is missing from disk")
        
    with open(job.result_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        
    return data

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    logger.debug("Health check endpoint hit")
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/v1/logs", response_class=PlainTextResponse)
async def view_live_logs(lines: int = 200):
    """View the last N lines of the live extraction logs in your browser."""
    log_path = r"C:\Users\DeepakTM\Music\Projects\lilly-pdf-extractor-agent\logs\agent_run.log"
    if not os.path.exists(log_path):
        return "Log file not found. Have you started a job yet?"
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            # Return only the last N lines
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"Error reading logs: {e}"

@app.get("/")
async def root():
    return {"message": "API is alive"}