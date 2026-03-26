import os
from dotenv import load_dotenv

load_dotenv()

DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# ==============================================================================
# LLM TIMEOUT & RETRY SETTINGS
# ==============================================================================
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
LLM_TIMEOUT_MULTIMODAL_SECONDS = int(os.getenv("LLM_TIMEOUT_MULTIMODAL_SECONDS", "240"))
LLM_RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "4"))

# ==============================================================================
# CONCURRENCY & THROTTLE SETTINGS
# ==============================================================================
MAX_CONCURRENT_PDFS = int(os.getenv("MAX_CONCURRENT_PDFS", "1"))
MAX_CONCURRENT_SECTIONS = int(os.getenv("MAX_CONCURRENT_SECTIONS", "3"))
COPY_HEADROOM = int(os.getenv("COPY_HEADROOM", "2"))
PAGE_THROTTLE_SECONDS = int(os.getenv("PAGE_THROTTLE_SECONDS", "3"))

# Auto-derived — change the knobs above, these cascade automatically
MAX_CONCURRENT_LLM_CALLS = MAX_CONCURRENT_SECTIONS * MAX_CONCURRENT_PDFS
PAGE_AGENT_COPIES = MAX_CONCURRENT_LLM_CALLS + COPY_HEADROOM
SEQUENTIAL_AGENT_COPIES = 3  # master/crop agents run sequentially, no scaling

# ==============================================================================
# PAGE-LEVEL RETRY (outer retry — wraps the entire page_agent invocation)
# ==============================================================================
PAGE_RETRY_ATTEMPTS = int(os.getenv("PAGE_RETRY_ATTEMPTS", "2"))
PAGE_RETRY_BACKOFF_SECONDS = int(os.getenv("PAGE_RETRY_BACKOFF_SECONDS", "60"))
