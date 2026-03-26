"""
Unified Cortex LLM wrapper — LangChain + dual auth (Azure AD / AWS STS).

Merges the best of cortex_llm_config.py and test_langchain_cortex.py:
 - Load‑balanced round‑robin model selection
 - Tenacity retry with exponential backoff
 - Async concurrency semaphore
 - Multimodal (image) support via OpenAI chat completions format
 - Structured output via LangChain with_structured_output (function_calling)
 - Manual schema‑injection fallback when function_calling is unsupported
 - Azure AD (MSAL/APIM) authentication for local dev
 - AWS STS (SigV4) authentication for production EKS
"""

import os
import json
import re
import threading
import itertools
import time
import logging
import asyncio
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()

from core.config import (
    LLM_TIMEOUT_SECONDS,
    LLM_TIMEOUT_MULTIMODAL_SECONDS,
    LLM_RETRY_ATTEMPTS,
    MAX_CONCURRENT_LLM_CALLS,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Types & Exceptions
# =============================================================================
T = TypeVar("T", bound=BaseModel)


class APIUnavailableError(Exception):
    pass


class JSONParsingError(Exception):
    pass


# =============================================================================
# Configuration — from environment
# =============================================================================
USE_AWS_AUTH = os.getenv("USE_AWS_AUTH", "").lower() in ("true", "1", "yes")

# Azure AD / APIM credentials (used when USE_AWS_AUTH is False)
CORTEX_CLIENT_ID = os.getenv("CORTEX_CLIENT_ID")
CORTEX_CLIENT_SECRET = os.getenv("CORTEX_CLIENT_SECRET")
CORTEX_AUTHORITY = os.getenv("CORTEX_AUTHORITY")
CORTEX_SCOPE = os.getenv("CORTEX_SCOPE")
CORTEX_BASE_URL = os.getenv(
    "CORTEX_BASE_URL",
    "https://gateway.apim.lilly.com/cortex/cortex-openai",
)

# =============================================================================
# Global concurrency limiter
# =============================================================================
_llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

# =============================================================================
# 1. AUTHENTICATION — Azure AD (MSAL)
# =============================================================================
_msal_app = None
_msal_lock = threading.Lock()


def _get_azure_ad_token() -> str:
    """Acquire an Azure AD bearer token via MSAL client‑credentials flow.

    Returns the raw token string (without "Bearer " prefix).
    Thread‑safe — MSAL app is lazily initialised behind a lock.
    """
    global _msal_app
    missing = [
        name
        for name, val in [
            ("CORTEX_CLIENT_ID", CORTEX_CLIENT_ID),
            ("CORTEX_CLIENT_SECRET", CORTEX_CLIENT_SECRET),
            ("CORTEX_AUTHORITY", CORTEX_AUTHORITY),
            ("CORTEX_SCOPE", CORTEX_SCOPE),
        ]
        if not val
    ]
    if missing:
        raise APIUnavailableError(
            f"Missing Azure AD env vars: {', '.join(missing)}. Set them in .env"
        )
    with _msal_lock:
        if _msal_app is None:
            import msal

            _msal_app = msal.ConfidentialClientApplication(
                client_id=CORTEX_CLIENT_ID,
                client_credential=CORTEX_CLIENT_SECRET,
                authority=CORTEX_AUTHORITY,
            )
    result = _msal_app.acquire_token_for_client(scopes=[CORTEX_SCOPE])
    if "access_token" not in result:
        error = result.get("error_description", result.get("error", "Unknown MSAL error"))
        raise APIUnavailableError(f"Azure AD token acquisition failed: {error}")
    return result["access_token"]


# =============================================================================
# 2. AUTHENTICATION — AWS STS (SigV4)
# =============================================================================
def _get_aws_auth_headers() -> dict:
    """Generate Cortex auth headers using AWS STS SigV4 signing.

    Production path — uses the boto3 credential provider chain
    (EKS IAM role, env vars, or ~/.aws/credentials).
    """
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    session = boto3.Session(region_name="us-east-1")
    request = AWSRequest(
        method="POST",
        url="https://sts.amazonaws.com/",
        data="Action=GetCallerIdentity&Version=2011-06-15",
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "Host": "sts.amazonaws.com",
        },
    )
    credentials = session.get_credentials()
    SigV4Auth(credentials, "sts", "us-east-1").add_auth(request)

    headers = {**request.headers}
    headers["User-Agent"] = "pharmai_client/1.0.0"
    headers["Accept"] = "application/json"
    headers["Content-Type"] = "application/json"
    headers.pop("Host", None)
    return headers


# =============================================================================
# 3. LANGCHAIN CHAT MODEL FACTORY
# =============================================================================
def _get_chat_model(model_name: str, temperature: float = 0, max_tokens: int = 40960):
    """Return a LangChain chat model configured for the active auth method.

    Azure AD  → AzureChatOpenAI  (token provider points at APIM gateway)
    AWS STS   → ChatOpenAI       (signed headers, openai‑compat endpoint)
    """
    if USE_AWS_AUTH:
        from langchain_openai import ChatOpenAI

        headers = _get_aws_auth_headers()
        base_url = CORTEX_BASE_URL.rstrip("/")
        if "/cortex-openai" not in base_url.lower():
            base_url += "/cortex-openai"

        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            default_headers=headers,
            api_key="placeholder",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=LLM_TIMEOUT_MULTIMODAL_SECONDS,
        )
    else:
        from langchain_openai import AzureChatOpenAI

        base_url = CORTEX_BASE_URL.rstrip("/")
        if "/cortex-openai" not in base_url.lower():
            base_url += "/cortex-openai"

        return AzureChatOpenAI(
            openai_api_version="2023-05-15",
            azure_endpoint=base_url,
            deployment_name=model_name,
            model=model_name,
            azure_ad_token_provider=_get_azure_ad_token,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=LLM_TIMEOUT_MULTIMODAL_SECONDS,
        )


# =============================================================================
# 4. THREAD‑SAFE ROUND‑ROBIN LOAD BALANCER
# =============================================================================
_rr_iterators: dict[tuple, itertools.cycle] = {}
_rr_lock = threading.Lock()


def get_next_model(model_configs: list[str]) -> str:
    """Thread‑safe round‑robin selection across a list of model deployment names."""
    group_key = tuple(model_configs)
    with _rr_lock:
        if group_key not in _rr_iterators:
            _rr_iterators[group_key] = itertools.cycle(model_configs)
        return next(_rr_iterators[group_key])


# =============================================================================
# 5. MESSAGE BUILDERS
# =============================================================================
def _build_messages(
    system_prompt: str,
    user_prompt: str,
    images_b64: Optional[str | list[str]] = None,
) -> list:
    """Build LangChain message objects (SystemMessage + HumanMessage).

    If images are supplied, the user message uses OpenAI multimodal content parts.
    """
    messages = []

    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))

    if images_b64:
        images_list = [images_b64] if isinstance(images_b64, str) else images_b64
        content_parts: list[dict] = [{"type": "text", "text": user_prompt}]
        for b64 in images_list:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        messages.append(HumanMessage(content=content_parts))
    else:
        messages.append(HumanMessage(content=user_prompt))

    return messages


# =============================================================================
# 6. SCHEMA INJECTION (fallback when function_calling is unsupported)
# =============================================================================
def _build_schema_repr(schema_obj: dict, definitions: Optional[dict] = None) -> dict:
    """Recursively build a human‑readable schema representation."""
    if definitions is None:
        definitions = schema_obj.get("$defs", {})

    if schema_obj.get("type") == "object":
        result = {}
        for field, props in schema_obj.get("properties", {}).items():
            desc = props.get("description", "")
            field_type = props.get("type", "")

            if field_type == "array" and "items" in props:
                items = props["items"]
                if "$ref" in items:
                    ref_name = items["$ref"].split("/")[-1]
                    result[field] = [_build_schema_repr(definitions.get(ref_name, {}), definitions)]
                elif items.get("type") == "object":
                    result[field] = [_build_schema_repr(items, definitions)]
                else:
                    result[field] = [f"<{items.get('type', 'any')}> {items.get('description', '')}"]
            elif "$ref" in props:
                ref_name = props["$ref"].split("/")[-1]
                result[field] = _build_schema_repr(definitions.get(ref_name, {}), definitions)
            else:
                result[field] = f"<{field_type}> {desc}" if desc else f"<{field_type}>"
        return result
    return schema_obj


def _inject_schema_into_prompt(system_prompt: str, response_model: Type[T]) -> str:
    """Append a JSON schema block to the system prompt so the LLM outputs valid JSON."""
    schema_dict = response_model.model_json_schema()
    example_structure = _build_schema_repr(schema_dict)
    schema_str = json.dumps(example_structure, indent=2)

    return system_prompt + (
        "\n\n=========================================\n"
        "OUTPUT FORMAT INSTRUCTIONS:\n"
        "You must output ONLY a valid, raw JSON object.\n"
        "Do not include any text before or after the JSON.\n"
        "Do not wrap your response in ```json markdown blocks.\n"
        "Your JSON must perfectly match this schema:\n\n"
        f"{schema_str}\n"
        "=========================================\n"
    )


def _parse_json_response(raw_message: str, response_model: Type[T]) -> T:
    """Parse raw LLM text into a validated Pydantic model.

    Handles markdown wrappers, hallucinated schema wrappers, and
    unescaped control characters.
    """
    if not raw_message:
        raise JSONParsingError(
            "API returned an empty message! Safety filter tripped or Agent Orchestrator crashed."
        )

    # Strip markdown ```json ... ``` wrappers
    pattern = r"^```(?:json)?\s*(.*?)\s*```$"
    match = re.search(pattern, raw_message, re.DOTALL | re.IGNORECASE)
    cleaned_json = match.group(1) if match else raw_message
    cleaned_json = cleaned_json.strip("` \n")

    try:
        parsed_dict = json.loads(cleaned_json, strict=False)

        # Auto‑correct hallucination: {"properties": {...}}
        if "properties" in parsed_dict and isinstance(parsed_dict["properties"], dict):
            inner = parsed_dict["properties"]
            corrected = {}
            for k, v in inner.items():
                corrected[k] = v["value"] if isinstance(v, dict) and "value" in v else v
            parsed_dict = corrected

        return response_model.model_validate(parsed_dict)

    except json.JSONDecodeError as e:
        raise JSONParsingError(
            f"LLM did not return valid JSON syntax: {e}\nRaw Output:\n{raw_message}"
        )
    except ValidationError as e:
        error_msg = (
            f"Pydantic Validation Failed:\n{e}\n\n"
            f"Expected Schema: {response_model.model_json_schema()}\n\n"
            f"Raw Output:\n{raw_message}"
        )
        logger.error(f"Cortex LangChain: Schema Validation Error:\n{error_msg}")
        raise JSONParsingError(error_msg)


# =============================================================================
# 7. THE MAIN CORTEX WRAPPER
# =============================================================================
@retry(
    stop=stop_after_attempt(LLM_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((APIUnavailableError, JSONParsingError)),
    reraise=True,
)
async def cortex_call_llm(
    model_configs: str | list[str],
    system_prompt: str,
    user_prompt: str,
    images_b64: Optional[str | list[str]] = None,
    response_model: Optional[Type[T]] = None,
) -> str | T:
    """Call a Cortex LLM model via LangChain.

    Signature is a drop‑in replacement for the existing ``cortex_call_llm``
    in ``cortex_llm_config.py``.

    Parameters
    ----------
    model_configs : str or list[str]
        A single model deployment name, or a list for round‑robin load balancing.
    system_prompt : str
        System‑level instructions for the LLM.
    user_prompt : str
        The user turn content.
    images_b64 : str or list[str], optional
        Base64‑encoded JPEG image(s) for multimodal requests.
    response_model : Type[BaseModel], optional
        If provided, the LLM output is parsed and validated into this Pydantic model.
        Uses LangChain ``with_structured_output`` (function_calling) first, then
        falls back to manual schema injection + JSON parsing.

    Returns
    -------
    str or T
        Raw text when *response_model* is None, otherwise a validated Pydantic instance.
    """
    # ── 1. Load‑balanced model selection ──────────────────────────────────
    model_name = (
        get_next_model(model_configs)
        if isinstance(model_configs, list)
        else model_configs
    )
    logger.debug(f"Cortex LangChain: Selected model -> {model_name}")

    start_time = time.time()

    async with _llm_semaphore:
        # ── 2. Build chat model ───────────────────────────────────────────
        chat_model = _get_chat_model(model_name)

        # ── 3A. STRUCTURED OUTPUT (with_structured_output) ────────────────
        if response_model:
            try:
                structured_model = chat_model.with_structured_output(
                    response_model, method="function_calling"
                )
                messages = _build_messages(system_prompt, user_prompt, images_b64)
                result = await structured_model.ainvoke(messages)

                latency = time.time() - start_time
                logger.debug(
                    f"Cortex LangChain: {model_name} structured output OK in {latency:.2f}s"
                )
                return result

            except (NotImplementedError, TypeError, KeyError, ValidationError) as e:
                # Model doesn't support function_calling or parsing failed —
                # fall through to manual schema injection below.
                logger.warning(
                    f"Cortex LangChain: with_structured_output failed ({type(e).__name__}: {e}), "
                    "falling back to manual schema injection."
                )
            except Exception as e:
                # Check if this is a retryable HTTP error
                status = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None
                )
                if status and status in (429, 500, 502, 503, 504):
                    raise APIUnavailableError(f"API {status}: {e}")
                # For other unexpected errors, also fall through to fallback
                logger.warning(
                    f"Cortex LangChain: with_structured_output error ({type(e).__name__}: {e}), "
                    "falling back to manual schema injection."
                )

            # ── 3B. FALLBACK: manual schema injection ─────────────────────
            augmented_prompt = _inject_schema_into_prompt(system_prompt, response_model)
            messages = _build_messages(augmented_prompt, user_prompt, images_b64)

            ai_msg = await chat_model.ainvoke(messages)
            raw_message = ai_msg.content.strip()

            latency = time.time() - start_time
            logger.debug(
                f"Cortex LangChain: {model_name} fallback responded in {latency:.2f}s"
            )
            logger.debug(f"Cortex LangChain: Raw response:\n{raw_message}")

            return _parse_json_response(raw_message, response_model)

        # ── 4. RAW TEXT OUTPUT (no response_model) ────────────────────────
        messages = _build_messages(system_prompt, user_prompt, images_b64)
        ai_msg = await chat_model.ainvoke(messages)
        raw_message = ai_msg.content.strip()

        latency = time.time() - start_time
        logger.debug(
            f"Cortex LangChain: {model_name} responded in {latency:.2f}s"
        )
        logger.debug(f"Cortex LangChain: Raw response:\n{raw_message}")

        return raw_message
