import os
import re
import asyncio
import random

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agents import page_agent, crop_agent
from core.state import MasterState
from utils.pdf_parser import (
    download_pdf, analyze_pdf_content, merge_page_text, 
    clean_markdown_headers, force_normalized_top_header_section, 
    remove_top_level_header_synopsis
)
import fitz
import base64
from agents.cortex_langchain import cortex_call_llm
from agents.agent_registry import AGENT_REGISTRY
from core.state import PDFQualityCheck, SynopsisMapping, SectionMapping, SynopsisQualityCheck
from core.config import MAX_CONCURRENT_SECTIONS, PAGE_THROTTLE_SECONDS, PAGE_RETRY_ATTEMPTS, PAGE_RETRY_BACKOFF_SECONDS

from core.state import PDFQualityCheck, SynopsisMapping, SectionMapping, SynopsisQualityCheck
import logging

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ==========================================
# LOADING THE DYNAMIC SECTION AND SYNOPSIS PROMPTS 
# ==========================================
from pathlib import Path

SPONSORS_SECTION_PROMPT_DIR = Path("prompts/sponsors/section")
SPONSORS_SYNOPSIS_PROMPT_DIR = Path("prompts/sponsors/synopsis")

def load_sponsor_section_prompt(sponsor_name: str) -> str:
    prompt_file = SPONSORS_SECTION_PROMPT_DIR / f"{sponsor_name.lower()}.txt"

    if not prompt_file.exists():
        general_prompt_file = SPONSORS_SECTION_PROMPT_DIR / "general.txt"
        logger.info(f"Section System Prompt not found: {sponsor_name.lower()}; so using general prompt...")
        return general_prompt_file.read_text()
    return prompt_file.read_text()

def load_sponsor_synopsis_prompt(sponsor_name: str) -> str:
    prompt_file = SPONSORS_SYNOPSIS_PROMPT_DIR / f"{sponsor_name.lower()}.txt"

    if not prompt_file.exists():
        general_prompt_file = SPONSORS_SYNOPSIS_PROMPT_DIR / "general.txt"
        logger.info(f"Synopsis System Prompt not found: {sponsor_name.lower()}; so using general prompt...")
        return general_prompt_file.read_text()
    return prompt_file.read_text()

async def ensure_pdf_exists(state: MasterState) -> str:
    """Helper to ensure the PDF exists on disk, re-downloading if necessary."""
    pdf_path = state.get("pdf_path")
    
    # If path is missing or file doesn't exist, re-calculate and re-download
    if not pdf_path or not os.path.exists(pdf_path):
        job_id = state.get("job_id", "fallback")
        cache_dir = "memory/pdf_cache"
        os.makedirs(cache_dir, exist_ok=True)
        pdf_path = os.path.join(cache_dir, f"{job_id}.pdf")
        
        if not os.path.exists(pdf_path):
            logger.info(f"📂 orchestrator: PDF missing at {pdf_path}. Re-downloading from {state['pdf_url']}...")
            download_pdf(state["pdf_url"], target_path=pdf_path)
        else:
            logger.info(f"📂 orchestrator: PDF found in cache at {pdf_path}.")
            
    return pdf_path

async def node_extract_toc(state: MasterState) -> MasterState:
    logger.info("🌐 Orchestrator: Downloading PDF and Extracting TOC...")
    
    # Use persistent cache instead of tempfile
    pdf_path = await ensure_pdf_exists(state)
    
    doc = fitz.open(pdf_path)
    toc_text, toc_image_list = analyze_pdf_content(doc)
    doc.close()
    
    logger.debug(f"🌐 [STATE DUMP] node_extract_toc completed. State Keys: {list(state.keys()) + ['pdf_path', 'toc_text', 'toc_images_b64']}")
    return {"pdf_path": pdf_path, "toc_text": toc_text, "toc_images_b64": toc_image_list}

async def node_validate_pdf_quality(state: MasterState) -> MasterState:
    logger.info("🌐 Orchestrator: Validating PDF Quality (Circuit Breaker)...")

    sys_prompt = "Compare extracted TOC text to the image. If text is severely fragmented with random spaces (e.g. 'C l i n i c a l'), return FAIL."
    user_prompt = f"<text>\n{state['toc_text']}\n</text>\nIs this corrupted?"

    pdf_quality_agents = AGENT_REGISTRY["masteragent-pdf-quality-validator"]
    result: PDFQualityCheck = await cortex_call_llm(
        model_configs=pdf_quality_agents,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        images_b64=state["toc_images_b64"],
        response_model=PDFQualityCheck
    )

    if result.status == "FAIL": logger.error(f"🚨 PIPELINE ABORTED: {result.reason}")
    else: logger.info("✅ PDF Quality Passed.")
    logger.debug(f"🌐 [STATE DUMP] node_validate_pdf_quality completed. State Keys: {list(state.keys()) + ['pdf_status']}")
    return {
        "pdf_status": result.status,
        "abort_node": "validate" if result.status == "FAIL" else "",
        "abort_reason": result.reason if result.status == "FAIL" else "",
    }

def check_pdf_quality(state: MasterState):
    return "end" if state.get("pdf_status") == "FAIL" else "continue"

async def node_map_synopsis_from_toc(state: MasterState) -> MasterState:
    logger.info("🌐 Orchestrator: Mapping Synopsis Pages...")

    sys_prompt = "Find the 'Synopsis' or 'Executive Summary' in the TOC and output its page range list."
    user_prompt = state['toc_text']

    synopsis_from_toc_agents = AGENT_REGISTRY["masteragent-synopsis-from-toc"]

    result: SynopsisMapping = await cortex_call_llm(
        model_configs=synopsis_from_toc_agents,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        response_model=SynopsisMapping
    )

    logger.debug(f"🌐 [STATE DUMP] node_map_synopsis_from_toc completed. State Keys: {list(state.keys()) + ['synopsis_pages']}")
    return {"synopsis_pages": result.synopsis_pages}

async def node_map_sections_from_toc(state: MasterState) -> MasterState:
    logger.info("🌐 Orchestrator: Mapping Required Normalized Sections from TOC...")

    sys_prompt = f"""You are an expert Clinical Protocol layout analyzer. 
Your goal is to map the actual sections found in the Table of Contents (TOC) to a specific list of REQUIRED STANDARD SECTIONS.

REQUIRED STANDARD SECTIONS:
{state['required_normalized_sections']}

CRITICAL RULES FOR MAPPING:
1. MUTUALLY EXCLUSIVE MAPPING: An actual PDF section can only be mapped ONCE. Do not map the same actual section to multiple standard sections.
2. COMPOUND SECTIONS: Clinical protocols often combine topics. If the PDF combines "Design" and "Population" into a single section (e.g., "Description of Design and Trial Population"):
   - DO map it to the compound standard section: "study design and population".
   - DO NOT map it to the individual "study design" or "study population" standard sections.
3. SPLIT SECTIONS: If the PDF separates the topics (e.g., "3.0 Study Design" and "4.0 Patient Selection"):
   - DO map them to the individual "study design" and "study population" standard sections.
   - DO NOT use the compound "study design and population" standard section.
4. EXACT TOP-LEVEL MATCHES: Map the main top-level section (e.g., 3.0). Do not map sub-sections (like 3.1, 3.2).
5. MISSING SECTIONS: If a standard section is completely missing or irrelevant, skip it. Do not force a bad match.

FEW-SHOT EXAMPLES:
Example A (Compound PDF):
TOC has: "3. Description of Design and Trial Population (Page 19)"
Correct Output: Map "study design and population" -> "3. Description of Design and Trial Population". (Skip 'study design' and 'study population' entirely).

Example B (Split PDF):
TOC has: "3. Trial Design (Page 15)" and "4. Selection of Subjects (Page 20)"
Correct Output: Map "study design" -> "3. Trial Design" AND "study population" -> "4. Selection of Subjects".
"""

    user_prompt = f"<toc_text>\n{state['toc_text']}\n</toc_text>"

    section_from_toc_agents = AGENT_REGISTRY["masteragent-sections-from-toc"]
    result: SectionMapping = await cortex_call_llm(
        model_configs=section_from_toc_agents,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        response_model=SectionMapping
    )
    
    # Log the results to the console so you can see the mapping
    logger.info("✅ Successfully Mapped Sections.")
    for s in result.sections:
         logger.debug(f"   - {s.normalized_title} -> Found as: '{s.actual_pdf_title}' (Pages {s.start_page}-{s.end_page})")

    # Store this rich list of objects in your state for the child agents to use
    logger.debug(f"🌐 [STATE DUMP] node_map_sections_from_toc completed. State Keys: {list(state.keys()) + ['section_pages']}")
    return {"section_pages": result.sections}

async def node_calculate_crop(state: MasterState) -> MasterState:
    logger.info("🌐 Orchestrator: Invoking Multimodal Crop Sub-Agent...")
    pdf_path = await ensure_pdf_exists(state)
    doc = fitz.open(pdf_path)
    
    # Pool synopsis pages plus all section page ranges collected from the TOC mapping
    all_target_pages = set(state["synopsis_pages"])

    for mapped_section in state["section_pages"]:
        start = mapped_section.start_page
        end = mapped_section.end_page
        if start and end and start <= end:
            all_target_pages.update(range(start, end + 1))

    # Safely pick up to 3 random pages
    target_list = list(all_target_pages)
    sample_pages = random.sample(target_list, min(3, len(target_list)))
    
    result = await crop_agent.graph.ainvoke({
        "pdf_path": state["pdf_path"],
        "sponsor_name": state["sponsor_name"],
        "sample_pages": sample_pages, # Pass the 3 pages here
        "historical_boxes": [],
        "spatial_json": "",
        "test_page_images_b64": [],
        "proposed_box": (0,0,0,0),
        "validation_reasoning": "",
        "is_valid": False,
        "iteration_count": 0,
        "memory_index_tested": 0

    }, config={"run_name": f"Crop Agent: {state['sponsor_name']}"})
    
    if result["is_valid"]:
        final_box = result["proposed_box"]
    else:
        logger.warning("Crop Agent failed. Falling back to Full Page to prevent data loss.")
        doc = fitz.open(state["pdf_path"])
        final_box = (0, 0, doc[0].rect.width, doc[0].rect.height)
        doc.close()
        
    logger.debug(f"🌐 [STATE DUMP] node_calculate_crop completed. State Keys: {list(state.keys()) + ['crop_box']}")
    return {"crop_box": final_box}

async def node_ingest_pdf(state: MasterState) -> MasterState:
    logger.info("🌐 Orchestrator: Ingesting mapped pages and applying dynamic crop...")
    pdf_path = await ensure_pdf_exists(state)
    doc = fitz.open(pdf_path)
    crop_rect = fitz.Rect(state["crop_box"])
    
    all_pages = set(state["synopsis_pages"])

    for mapped_section in state["section_pages"]:
        start = mapped_section.start_page
        end = mapped_section.end_page
        if start and end and start <= end:
            all_pages.update(range(start, end + 1))
        
    raw_pages = {}
    for page_num in all_pages:
        # Prevent out-of-bounds page requests
        if page_num - 1 >= len(doc): continue 
        page = doc[page_num - 1]
        
        # EXCLUDE HEADER/FOOTER BY CLIPPING
        text = page.get_text("text", clip=crop_rect)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop_rect) 
        raw_pages[page_num] = {"image_b64": base64.b64encode(pix.tobytes("jpeg")).decode('utf-8'), "text": text}
        
    doc.close()
    logger.debug(f"🌐 [STATE DUMP] node_ingest_pdf completed. State Keys: {list(state.keys()) + ['raw_pages']}")
    return {"raw_pages": raw_pages}

async def node_process_synopsis(state: MasterState) -> MasterState:
    logger.info("🌐 Orchestrator: Processing Synopsis Pages...")

    base_sys_prompt = load_sponsor_synopsis_prompt(state["sponsor_name"])

    accumulated_text = ""
    cleaned_parts = []
    for page_num in state["synopsis_pages"]:
        if page_num not in state["raw_pages"]: continue
        
        page_data = state["raw_pages"][page_num]
        result = None
        for attempt in range(1, PAGE_RETRY_ATTEMPTS + 1):
            try:
                result = await page_agent.graph.ainvoke({
                    "page_num": page_num,
                    "mode": "synopsis",
                    "system_prompt": base_sys_prompt,
                    "target_section": "",
                    "image_b64": page_data["image_b64"],
                    "pymupdf_text": page_data["text"],
                    "previous_context": "",
                    "iteration_count": 0,
                    "validation_feedback": [],
                    "attempt_history": [],
                    "page_failed": False
                })
                break  # success — exit retry loop
            except Exception as e:
                logger.warning(
                    f"   -> Synopsis page {page_num} failed (attempt {attempt}/{PAGE_RETRY_ATTEMPTS}): "
                    f"{type(e).__name__}: {e}"
                )
                if attempt < PAGE_RETRY_ATTEMPTS:
                    logger.info(f"   -> Waiting {PAGE_RETRY_BACKOFF_SECONDS}s before retrying synopsis page {page_num}...")
                    await asyncio.sleep(PAGE_RETRY_BACKOFF_SECONDS)
                else:
                    logger.error(f"   -> Synopsis page {page_num} SKIPPED after {PAGE_RETRY_ATTEMPTS} attempts.")

        if result is None:
            continue

        extracted_text = result.get("final_markdown", "").strip()
        
        if extracted_text:
            # Dynamically merges
            if not accumulated_text:
                accumulated_text = extracted_text
            else:
                accumulated_text = merge_page_text(accumulated_text, extracted_text)

        cleaned_parts.append(result["final_markdown"])
        await asyncio.sleep(PAGE_THROTTLE_SECONDS)
        
    logger.debug(f"🌐 [STATE DUMP] node_process_synopsis completed. State Keys: {list(state.keys()) + ['cleaned_synopsis']}")
    return {"cleaned_synopsis": accumulated_text}

async def node_check_synopsis_sufficiency(state: MasterState) -> MasterState:
    logger.info("🌐 Orchestrator: Checking if Synopsis contains sufficient data...")
    
    sys_prompt = "You are a Clinical Data Validator. Analyze the text. Does it contain actual Clinical Trial Synopsis data (Objectives, Methodology, etc.)? If it is just a title page or missing core data, return FAIL."
    user_prompt = state["cleaned_synopsis"]

    pdf_quality_agents = AGENT_REGISTRY["masteragent-synopsis-check"]

    result: SynopsisQualityCheck = await cortex_call_llm(
        model_configs=pdf_quality_agents,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        response_model=SynopsisQualityCheck
    )

    if result.status == "FAIL":
        logger.error(f"🚨 PIPELINE ABORTED: Synopsis data is insufficient. Reason: {result.reason}")
    logger.debug(f"🌐 [STATE DUMP] node_check_synopsis_sufficiency completed. State Keys: {list(state.keys()) + ['pdf_status']}")
    return {
        "pdf_status": result.status,
        "abort_node": "check_synopsis" if result.status == "FAIL" else "",
        "abort_reason": result.reason if result.status == "FAIL" else "",
    }

async def process_single_section(mapped_section, state: MasterState, base_sys_prompt: str) -> tuple[str, str]:
    norm_title = mapped_section.normalized_title
    actual_title = mapped_section.actual_pdf_title
    start_page = mapped_section.start_page
    end_page = mapped_section.end_page
    
    # 2. Safety check for valid page ranges
    if not start_page or not end_page or start_page > end_page:
        logger.warning(f"⚠️ Invalid page range for {norm_title} ({start_page}-{end_page}). Skipping.")
        return norm_title, ""

    # 3. Build the page numbers list (inclusive of the end_page)
    pages_to_process = range(start_page, end_page + 1)
    
    logger.info(f"Processing Section: '{norm_title}'")
    logger.debug(f"   -> Found as '{actual_title}' on Pages {start_page} to {end_page}")
    
    accumulated_text = ""
    previous_tail = ""

    # 4. Inject the ACTUAL PDF title into the system prompt
    sys_prompt = base_sys_prompt.replace("{target_section_title}", actual_title)
    
    for page_num in pages_to_process:
        if page_num not in state["raw_pages"]: 
            logger.warning(f"   -> Page {page_num} not found in raw_pages. Skipping.")
            continue
            
        page_data = state["raw_pages"][page_num]
        
        logger.debug(f"   -> Extractor Agent running on page {page_num}...")
        
        # 5. Invoke the Page Agent with page-level retry
        # NOTE: We pass 'actual_title' as the target_section so the VLM knows exactly what words to look for on the page.
        res = None
        for attempt in range(1, PAGE_RETRY_ATTEMPTS + 1):
            try:
                res = await page_agent.graph.ainvoke({
                    "page_num": page_num, 
                    "mode": "section",
                    "system_prompt": sys_prompt,
                    "target_section": actual_title, 
                    "image_b64": page_data["image_b64"], 
                    "pymupdf_text": page_data["text"],
                    "previous_context": previous_tail, 
                    "iteration_count": 0,
                    "validation_feedback": [],
                    "attempt_history": [],
                    "page_failed": False
                })
                break  # success — exit retry loop
            except Exception as e:
                logger.warning(
                    f"   -> Page {page_num} in section '{norm_title}' failed (attempt {attempt}/{PAGE_RETRY_ATTEMPTS}): "
                    f"{type(e).__name__}: {e}"
                )
                if attempt < PAGE_RETRY_ATTEMPTS:
                    logger.info(f"   -> Waiting {PAGE_RETRY_BACKOFF_SECONDS}s before retrying page {page_num}...")
                    await asyncio.sleep(PAGE_RETRY_BACKOFF_SECONDS)
                else:
                    logger.error(f"   -> Page {page_num} in section '{norm_title}' SKIPPED after {PAGE_RETRY_ATTEMPTS} attempts.")
        
        if res is None:
            continue
        
        extracted_text = res.get("final_markdown", "").strip()
        
        if extracted_text:
            # Dynamically merge
            if not accumulated_text:
                accumulated_text = extracted_text
            else:
                accumulated_text = merge_page_text(accumulated_text, extracted_text)
            
            # Keep the last 1000 characters for the sliding window context
            previous_tail = extracted_text[-1000:]
        
        # Throttle to prevent 503 errors from LLM APIs
        await asyncio.sleep(PAGE_THROTTLE_SECONDS)
        
    return norm_title, accumulated_text


async def node_process_sections(state: MasterState) -> MasterState:
    logger.info("🌐 Orchestrator: Processing Sections Conditionally and Concurrently...")
    cleaned_sections = {}

    # Load the base prompt for the specific sponsor
    base_sys_prompt = load_sponsor_section_prompt(state["sponsor_name"])

    # Semaphore to limit concurrent section processing and reduce API pressure
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SECTIONS)

    async def guarded_section(mapped_section):
        async with semaphore:
            return await process_single_section(mapped_section, state, base_sys_prompt)

    tasks = []
    for mapped_section in state["section_pages"]:
        tasks.append(guarded_section(mapped_section))
            
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"🌐 Section processing failed: {type(result).__name__}: {result}")
            continue
        norm_title, text = result
        cleaned_sections[norm_title] = text

    logger.debug(f"🌐 [STATE DUMP] node_process_sections completed. State Keys: {list(state.keys()) + ['cleaned_sections']}")
    return {"cleaned_sections": cleaned_sections}

async def node_assemble_dataset(state: MasterState) -> MasterState:
    logger.info("🌐 Orchestrator: Assembling final JSON dataset...")
    final_dataset = {}

    # Clean the input (Synopsis) by removing its H1 title
    clean_input_body = remove_top_level_header_synopsis(state["cleaned_synopsis"])
    
    for norm_title, body in state["cleaned_sections"].items():
        # Skip empty sections
        if not body.strip():
            continue
            
        # 1. Clean the section numbers from all headers in the body
        body_cleaned = clean_markdown_headers(body)
        
        # 2. Force the top-level header to be the exact Normalized Title
        final_output_body = force_normalized_top_header_section(body_cleaned, norm_title)
        
        # Format the title nicely for the instruction (e.g., "study design" -> "Study Design")
        pretty_title = norm_title.title()
        
        # 3. Build the instruction using the Normalized Title
        instruction = (
            "You are an expert Clinical Medical Writer. Below is the Synopsis for a clinical trial.\n"
            f"Based strictly on this information, generate the content for the section: {pretty_title}."
        )
        
        final_dataset[norm_title] = {
            "instruction": instruction,
            "input": clean_input_body,
            "output": final_output_body.strip()
        }
        
    logger.debug(f"🌐 [STATE DUMP] node_assemble_dataset completed. State Keys: {list(state.keys()) + ['final_dataset']}")
    return {"final_dataset": final_dataset}

# --- BUILD MASTER GRAPH ---
workflow = StateGraph(MasterState)
workflow.add_node("extract_toc", node_extract_toc)
workflow.add_node("validate", node_validate_pdf_quality)
workflow.add_node("check_synopsis", node_check_synopsis_sufficiency)
workflow.add_node("map_synopsis", node_map_synopsis_from_toc)
workflow.add_node("map_sections", node_map_sections_from_toc)
workflow.add_node("calculate_crop", node_calculate_crop)
workflow.add_node("ingest", node_ingest_pdf)
workflow.add_node("synopsis", node_process_synopsis)
workflow.add_node("sections", node_process_sections)
workflow.add_node("assemble", node_assemble_dataset)

# Routing Flow
workflow.set_entry_point("extract_toc")
workflow.add_conditional_edges("validate", check_pdf_quality, {"end": END, "continue": "map_synopsis"})

workflow.add_edge("extract_toc", "validate")
workflow.add_edge("map_synopsis", "map_sections")
workflow.add_edge("map_sections", "calculate_crop")
workflow.add_edge("calculate_crop", "ingest")
workflow.add_edge("ingest", "synopsis")
workflow.add_edge("synopsis", "check_synopsis")
workflow.add_conditional_edges("check_synopsis", check_pdf_quality, {"end": END, "continue": "sections"})
workflow.add_edge("sections", "assemble")
workflow.add_edge("assemble", END)

if __name__ == "__main__":

    sponsor_name = "Boehringer Ingelheim"
    prompt = load_sponsor_synopsis_prompt(sponsor_name)
    if prompt:
        print(prompt)
    else:
        print("prompt wasn't extracted")
