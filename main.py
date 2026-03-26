import json
import re
import os
import uuid
import asyncio
from datetime import datetime, timezone
from utils.title_card import title

# 1. NEW: Import workflow instead of master_app, and import AsyncSqliteSaver
from agents.master_graph import workflow
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from core.initial_state import create_initial_state
from core.logger import setup_logger

logger = setup_logger()

async def run_agent():
    print(title)
    logger.info("🚀 Starting Fully Autonomous Clinical Agent...")
    
    test_pdf_url = "https://cdn.clinicaltrials.gov/large-docs/85/NCT02751385/Prot_000.pdf" 
    match = re.search(r"NCT\d+", test_pdf_url)
    nct_number = match.group() if match else "UNKNOWN_NCT"
    
    run_id = str(uuid.uuid4())
    
    initial_state = create_initial_state(
        job_id=run_id,
        pdf_url=test_pdf_url,
        sponsor_name="Boehringer Ingelheim"
    )

    # 2. NEW: Wrap the execution in the AsyncSqliteSaver context manager
    async with AsyncSqliteSaver.from_conn_string("memory/master_graph_checkpoints.db") as memory:
        
        # Compile the app right here with the active async memory connection!
        master_app = workflow.compile(checkpointer=memory)

        result_state = await master_app.ainvoke(
            initial_state, 
            config={
                "configurable": {"thread_id": run_id},
                "run_name": f"Extraction Pipeline: {nct_number}"
            }
        )
    
    if result_state.get("pdf_status") == "FAIL":
        abort_node = result_state.get("abort_node", "unknown")
        abort_reason = result_state.get("abort_reason", "No reason captured")
        failure_payload = {
            "job_id": run_id,
            "pdf_url": test_pdf_url,
            "sponsor_name": "Boehringer Ingelheim",
            "status": "ABORTED",
            "abort_node": abort_node,
            "abort_reason": abort_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            from utils.cloud_storage import storage_manager
            abort_filename = f"aborted/{nct_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ABORTED.json"
            storage_manager.upload_json(abort_filename, failure_payload)
            logger.info(f"☁️ Abort status uploaded to GCP -> {abort_filename}")
        except Exception as cloud_e:
            logger.warning(f"⚠️ Could not upload abort status to GCP: {cloud_e}")
        logger.error(f"❌ Process halted at node '{abort_node}': {abort_reason}")
    else:
        final_json = result_state["final_dataset"]
        logger.info("✅ PROCESS COMPLETE. Saving to JSON...")
        
        output_folder = "output_json"
        os.makedirs(output_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{nct_number}_{timestamp}.json"
        file_path = os.path.join(output_folder, file_name)
        
        with open(file_path, "w", encoding="utf-8") as f:
             json.dump(final_json, f, indent=4, ensure_ascii=False)
             
        logger.info(f"📁 File successfully saved at: {file_path}")

if __name__ == "__main__":
    asyncio.run(run_agent())