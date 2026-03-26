import logging
import sys
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from core.config import DEBUG_MODE

def setup_logger():
    os.makedirs("logs", exist_ok=True)
    
    # Grab the ROOT logger (this catches EVERYTHING in the app)
    root_logger = logging.getLogger()

    # Log folder
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # dynamic timestamp for log file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    info_log_path = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    # Only configure if we haven't already (prevents duplicate log lines)
    if not root_logger.handlers:
        if DEBUG_MODE:
            root_logger.setLevel(logging.DEBUG)
        else:
            root_logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 1. Output to Console (Terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # 2. Output to File
        file_handler = logging.FileHandler(info_log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # 3. Force Uvicorn to write to our file too
        for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
            ext_logger = logging.getLogger(logger_name)
            ext_logger.handlers = [] # Clear uvicorn's default handlers
            ext_logger.propagate = True # Send it up to the root logger
        # Silence noisy third-party loggers
        for noisy in ("aiosqlite", "urllib3", "httpx", "langsmith", "httpcore", "hpack"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        # Per-module DEBUG override (works even when root logger is INFO)
        # Set DEBUG_AGENTS=true in .env to enable without turning on global DEBUG
        if os.getenv("DEBUG_AGENTS", "false").lower() == "true" or DEBUG_MODE:
            for agent_module in ("agents.cortex_langchain", "agents.page_agent", "agents.master_graph"):
                logging.getLogger(agent_module).setLevel(logging.DEBUG)

    # Return a specific named logger for the script that called it
    return logging.getLogger("clinical_agent")