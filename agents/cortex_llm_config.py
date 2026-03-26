import os
import json
import re
import base64
import httpx
import threading
import itertools
from typing import Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from PIL import Image
import io
import time
import logging

from dotenv import load_dotenv
load_dotenv()

import asyncio
from core.config import LLM_TIMEOUT_SECONDS, LLM_TIMEOUT_MULTIMODAL_SECONDS, LLM_RETRY_ATTEMPTS, MAX_CONCURRENT_LLM_CALLS

logger = logging.getLogger(__name__)

# Global semaphore to limit total in-flight LLM API calls across all jobs
_llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
# ==============================================================================
# 1. UNIVERSAL CORTEX API WRAPPER
# ==============================================================================

T = TypeVar("T", bound=BaseModel)

class APIUnavailableError(Exception): pass
class JSONParsingError(Exception): pass

# You can fetch these from environment variables or a secure vault
CORTEX_COOKIE = os.getenv("CORTEX_COOKIE", None)
CORTEX_API_KEY = os.getenv("CORTEX_API_KEY", None)

# ==============================================================================
# IMAGE STITCHER (The Multi-Page Hack)
# ==============================================================================
def stitch_images_vertically(base64_list: list[str]) -> str:
    """Combines multiple base64 images into a single vertical base64 image."""
    if not base64_list:
        return ""
    if len(base64_list) == 1:
        return base64_list[0]

    # Decode base64 strings into PIL Images
    images = [Image.open(io.BytesIO(base64.b64decode(b64))) for b64 in base64_list]
    
    # Calculate the size of the new stitched image
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)

    # Create a blank canvas and paste the images one under the other
    stitched_image = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for img in images:
        stitched_image.paste(img, (0, y_offset))
        y_offset += img.size[1]

    # Convert back to base64
    buffered = io.BytesIO()
    stitched_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# ==============================================================================
# THREAD-SAFE ROUND-ROBIN LOAD BALANCER
# ==============================================================================
_rr_iterators = {}
_rr_lock = threading.Lock()

def get_next_model(model_configs: list[str]) -> str:
    """Thread-safe Round-Robin selection for a list of models."""
    # Create a unique key for this specific group of agents
    group_key = tuple(model_configs)
    
    with _rr_lock:  # Lock prevents race conditions during concurrent LangGraph batching
        if group_key not in _rr_iterators:
            _rr_iterators[group_key] = itertools.cycle(model_configs)
        return next(_rr_iterators[group_key])

# ==============================================================================
# THE CORTEX WRAPPER
# ==============================================================================

@retry(
    stop=stop_after_attempt(LLM_RETRY_ATTEMPTS), 
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((APIUnavailableError, JSONParsingError, httpx.RequestError)),
    reraise=True
)
async def cortex_call_llm(
    model_configs: str | list[str],
    system_prompt: str,
    user_prompt: str,
    images_b64: Optional[str | list[str]] = None,
    response_model: Optional[Type[T]] = None,
) -> str | T:
    
    # 1. LOAD BALANCER
    model_name = get_next_model(model_configs) if isinstance(model_configs, list) else model_configs
    logger.debug(f"Cortex Config: Selected Load-Balanced Model -> {model_name}")
    url_base = "https://cortex.lilly.com"
    
    # 2. AUTHENTICATION
    headers = {"accept": "application/json"}
    if CORTEX_COOKIE:
        headers["cookie"] = CORTEX_COOKIE
    elif CORTEX_API_KEY:
        headers["Authorization"] = f"Bearer {CORTEX_API_KEY}"

    # 3. JSON SCHEMA INJECTION (LangChain Style)
    if response_model:
        schema_dict = response_model.model_json_schema()
        
        # Function to recursively build a clearer schema representation
        def build_schema_repr(schema_obj, definitions=None):
            """Recursively build a human-readable schema representation."""
            if definitions is None:
                definitions = schema_dict.get("$defs", {})
            
            if schema_obj.get("type") == "object":
                result = {}
                for field, props in schema_obj.get("properties", {}).items():
                    desc = props.get("description", "")
                    field_type = props.get("type", "")
                    
                    if field_type == "array" and "items" in props:
                        items = props["items"]
                        if "$ref" in items:
                            # Reference to a definition - expand it
                            ref_name = items["$ref"].split("/")[-1]
                            ref_schema = definitions.get(ref_name, {})
                            result[field] = [build_schema_repr(ref_schema, definitions)]
                        elif items.get("type") == "object":
                            result[field] = [build_schema_repr(items, definitions)]
                        else:
                            result[field] = [f"<{items.get('type', 'any')}> {items.get('description', '')}"]
                    elif "$ref" in props:
                        ref_name = props["$ref"].split("/")[-1]
                        ref_schema = definitions.get(ref_name, {})
                        result[field] = build_schema_repr(ref_schema, definitions)
                    else:
                        result[field] = f"<{field_type}> {desc}" if desc else f"<{field_type}>"
                return result
            return schema_obj
        
        example_structure = build_schema_repr(schema_dict)
        schema_str = json.dumps(example_structure, indent=2)

        system_prompt += (
            f"\n\n=========================================\n"
            f"OUTPUT FORMAT INSTRUCTIONS:\n"
            f"You must output ONLY a valid, raw JSON object.\n"
            f"Do not include any text before or after the JSON.\n"
            f"Do not wrap your response in ```json markdown blocks.\n"
            f"Your JSON must perfectly match this schema:\n\n"
            f"{schema_str}\n"
            f"=========================================\n"
        )
        logger.debug(f"Cortex Config: Enforcing target JSON schema structure:\n{schema_str}")

    combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    logger.debug(f"Cortex Config: Outbound API Payload:\n{combined_prompt}")

    start_time = time.time()
    timeout = httpx.Timeout(
        connect=30.0,
        read=LLM_TIMEOUT_MULTIMODAL_SECONDS if images_b64 else LLM_TIMEOUT_SECONDS,
        write=30.0,
        pool=30.0
    )
    async with _llm_semaphore:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        
            # --- MULTIMODAL ROUTE (Images) ---
            if images_b64:
                url = f"{url_base}/model/ask/{model_name}"
                params = {"stream": "false", "use_responses_api": "false"}
                
                images_list = [images_b64] if isinstance(images_b64, str) else images_b64
                data = {"q": combined_prompt}
                
                if len(images_list) == 1:
                    # Single image: use uploaded_file (singular)
                    image_bytes = base64.b64decode(images_list[0])
                    files = {"uploaded_file": ("image.jpg", image_bytes, "image/jpeg")}
                    response = await client.post(url, headers=headers, params=params, data=data, files=files)
                else:
                    # Multiple images: use uploaded_files (plural) with separate entries
                    files = [("uploaded_files", (f"image{i+1}.jpg", base64.b64decode(b64), "image/jpeg")) for i, b64 in enumerate(images_list)]
                    response = await client.post(url, headers=headers, params=params, data=data, files=files)
                
            # --- PURE TEXT ROUTE ---
            else:
                url = f"{url_base}/model/ask/{model_name}"
                params = {"stream": "false", "use_responses_api": "false"}
                data = {"q": combined_prompt}
                files = {
                    "uploaded_file": (None, ""),
                }
                response = await client.post(url, headers=headers, params=params, data=data, files=files)

            # 4. ERROR HANDLING
            elapsed = time.time() - start_time
            if response.status_code in [500, 502, 503, 504, 429]:
                err_body = response.text[:500]
                logger.warning(
                    f"🔥 API {response.status_code} from '{model_name}' | URL: {url} | "
                    f"Elapsed: {elapsed:.1f}s | Response: {err_body}"
                )
                raise APIUnavailableError(f"API Error {response.status_code}: {err_body}")
            
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"🔥 HTTP {e.response.status_code} from '{model_name}' | URL: {url} | "
                    f"Elapsed: {elapsed:.1f}s | Response: {e.response.text[:500]}"
                )
                raise APIUnavailableError(f"HTTP Error: {e.response.text}")

            result = response.json()
            raw_message = result.get("message", "").strip()

    latency = time.time() - start_time
    logger.debug(f"Cortex Config: Model {model_name} responded successfully in {latency:.2f} seconds.")
    logger.debug(f"Cortex Config: Raw LLM Text Response:\n{raw_message}")

    # 5. JSON PARSING ROUTE
    if response_model:
        if not raw_message:
            raise JSONParsingError("API returned an empty message! Safety filter tripped or Agent Orchestrator crashed.")

        # Strip markdown wrappers just in case
        pattern = r"^```(?:json)?\s*(.*?)\s*```$"
        match = re.search(pattern, raw_message, re.DOTALL | re.IGNORECASE)
        cleaned_json = match.group(1) if match else raw_message
        cleaned_json = cleaned_json.strip("` \n")

        try:
            # Parse to a Python dictionary first, allowing unescaped control characters 
            parsed_dict = json.loads(cleaned_json, strict=False)
            
            # --- AUTO-CORRECT HALLUCINATIONS ---
            # If the LLM still wrapped it in {"properties": {...}} or {"title": ...}
            if "properties" in parsed_dict and isinstance(parsed_dict["properties"], dict):
                inner_dict = parsed_dict["properties"]
                
                # Sometimes it does: "scratchpad": {"value": "actual answer"}
                corrected_dict = {}
                for k, v in inner_dict.items():
                    if isinstance(v, dict) and "value" in v:
                        corrected_dict[k] = v["value"]
                    else:
                        corrected_dict[k] = v
                
                parsed_dict = corrected_dict
            # -----------------------------------

            return response_model.model_validate(parsed_dict)
            
        except json.JSONDecodeError as e:
            raise JSONParsingError(f"LLM did not return valid JSON syntax: {e}\nRaw Output:\n{raw_message}")
        except ValidationError as e:
            # Provide more detailed error info
            error_msg = f"Pydantic Validation Failed:\n{str(e)}\n\nExpected Schema: {response_model.model_json_schema()}\n\nRaw Output:\n{raw_message}"
            logger.error(f"Cortex Config: Schema Validation Error Details:\n{error_msg}")
            raise JSONParsingError(error_msg)

    return raw_message