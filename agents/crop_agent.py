import os
import fitz
import json
import base64
from pathlib import Path

from core.state import CropState, SuggestedCropBox, CropValidation
from agents.cortex_langchain import cortex_call_llm
from agents.agent_registry import AGENT_REGISTRY
from langgraph.graph import StateGraph, END
from utils.cloud_storage import storage_manager
import logging
from core.config import DEBUG_MODE

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

memory_dir = Path("memory")
MEMORY_FILE = memory_dir / "sponsor_based_crop_memory.json"

def load_memory() -> dict:
    cloud_mem = storage_manager.download_json("memory/sponsor_based_crop_memory.json")
    if cloud_mem is not None:
        return cloud_mem

    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory(sponsor: str, box: tuple):
    mem = load_memory()
    if sponsor not in mem: mem[sponsor] = []
    
    # Only append if it's a new unique box. Keep max 3 to prevent bloat.
    if box not in mem[sponsor]:
        mem[sponsor].insert(0, box) # Add to front (most recently used)
        mem[sponsor] = mem[sponsor][:3] 
        
    os.makedirs(memory_dir, exist_ok=True)
    with open(MEMORY_FILE, "w") as f: json.dump(mem, f, indent=4)
    storage_manager.upload_json("memory/sponsor_based_crop_memory.json", mem)

def render_page_with_bbox(pdf_path: str, page_num: int, box: tuple) -> str:
    """Draws a red bounding box on the page and returns base64 JPEG (in-memory, no file I/O)."""
    doc = fitz.open(pdf_path)
    if page_num - 1 >= len(doc):
        doc.close()
        return ""
    page = doc[page_num - 1]
    rect = fitz.Rect(box)
    page.draw_rect(rect, color=(1, 0, 0), width=2, fill_opacity=0.1)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_b64 = base64.b64encode(pix.tobytes("jpeg")).decode('utf-8')
    doc.close()
    return img_b64

def draw_and_save_bbox(pdf_path: str, page_num: int, box: tuple, iteration: int, sponsor_name: str):
    """Draws a red bounding box on the specified page and saves it as an image."""
    # Organize outputs by Sponsor Name
    output_dir = os.path.join("test_cropagent_vizualizer", sponsor_name.replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    if page_num - 1 >= len(doc):
        doc.close()
        return
        
    page = doc[page_num - 1] # 0-indexed
    
    # Draw a Red Rectangle (RGB: 1, 0, 0)
    rect = fitz.Rect(box)
    page.draw_rect(rect, color=(1, 0, 0), width=2, fill_opacity=0.1)
    
    # Render to Image and Save
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    out_file = os.path.join(output_dir, f"iter_{iteration}_bbox_page_{page_num}.png")
    pix.save(out_file)
    doc.close()
    
    logger.debug(f"🖼️ VISUALIZATION SAVED: {out_file}")

async def node_extract_and_load_memory(state: CropState) -> CropState:
    logger.info(f"✂️ CropAgent: Loading memory and extracting multimodal data for {state['sponsor_name']}...")
    
    # 1. Load historical boxes (List of up to 10 tuples)
    mem = load_memory()
    historical_boxes = mem.get(state["sponsor_name"], [])
    logger.debug(f"   -> Memory Check: Found {len(historical_boxes)} historical boxes for {state['sponsor_name']}.")
    
    # 2. Extract Spatial JSON and Images from ALL sample pages
    doc = fitz.open(state["pdf_path"])
    images_b64 = []
    spatial_data = []
    
    for p_num in state["sample_pages"]:
        if p_num - 1 >= len(doc):
            continue
        page = doc[p_num - 1]
        
        # Image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        images_b64.append(base64.b64encode(pix.tobytes("jpeg")).decode('utf-8'))
        
        # Spatial JSON (Only top/bottom/edges to save tokens)
        p_h, p_w = page.rect.height, page.rect.width
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, b_type, _ = b
            if b_type == 0 and len(text.strip()) > 3:
                if y1 < p_h*0.25 or y0 > p_h*0.75 or x1 < p_w*0.15 or x0 > p_w*0.85:
                    spatial_data.append({"page": p_num, "text": text.strip().replace("\n", ""), "y0": round(y0,1), "y1": round(y1,1), "x0": round(x0,1), "x1": round(x1,1)})
    
    doc.close()
    logger.debug(f"   -> Extracted {len(images_b64)} page images and {len(spatial_data)} spatial blocks.")
    
    return {
        "historical_boxes": historical_boxes, 
        "spatial_json": json.dumps(spatial_data, indent=2),
        "test_page_images_b64": images_b64,
        "memory_index_tested": 0,
        "is_valid": False,
        "attempt_history": []
    }

async def node_propose_crop(state: CropState) -> CropState:

    # 1. TEST HISTORY FIRST: If we have untested memory boxes, propose the next one!
    idx = state.get("memory_index_tested", 0)
    if idx < len(state["historical_boxes"]):
        logger.debug(f"   -> Testing historical memory box {idx+1}/{len(state['historical_boxes'])}...")
        return {
            "proposed_box": tuple(state["historical_boxes"][idx]), 
            "memory_index_tested": idx + 1,
            "current_proposal_reasoning": "Used historical memory box."
        }
    
    # 2. NO HISTORY LEFT: Ask the Multimodal LLM to generate a new one
    iteration = state.get("iteration_count", 0)
    logger.info(f"✂️ CropAgent: Asking VLM to propose new bounding box (Attempt {iteration+1})...")
    
    doc = fitz.open(state["pdf_path"])
    p_w, p_h = doc[0].rect.width, doc[0].rect.height
    doc.close()
    
    sys_prompt = f"""You are a Spatial Layout AI. Calculate a safe bounding box for the MAIN BODY text of a {state['sponsor_name']} document.
    Dimensions: Width={p_w}, Height={p_h}.
    
    You are given {len(state['sample_pages'])} sample pages from the document (pages {state['sample_pages']}).
    Look at ALL the images to visually identify where the headers/footers end.
    Then, use the SPATIAL JSON to find the exact coordinates of those blocks.
    Compare across all pages to find CONSISTENT header/footer boundaries.
    Return coordinates that EXCLUDE headers/footers/sidebars, but KEEP all body text.

    IMPORTANT - LEFT/RIGHT BREATHING SPACE:
    Do NOT make the bounding box tightly hug the text on the left and right sides.
    Add comfortable horizontal padding (at least 10-15 points) on BOTH the left and right edges beyond the outermost text.
    The left_x0 should be noticeably to the LEFT of where the leftmost body text starts.
    The right_x1 should be noticeably to the RIGHT of where the rightmost body text ends.
    This ensures no text is truncated and the crop has natural whitespace margins on both sides.
    """
    
    # NEW: INJECT CONVERSATION HISTORY (PAST MISTAKES)
    history = state.get("attempt_history", [])
    if history:
        sys_prompt += "\n\n⚠️ CRITICAL - PAST FAILED ATTEMPTS ⚠️\nDo NOT propose these exact coordinates again. Learn from these mistakes:\n"
        for i, attempt in enumerate(history):
            sys_prompt += f"- Attempt {i+1} Coordinates: {attempt['box']}\n"
            sys_prompt += f"  Your Previous Reasoning: {attempt.get('propose_reasoning', 'N/A')}\n"
            sys_prompt += f"  Validation Failure Reason: {attempt['val_reasoning']}\n"
            sys_prompt += f"  ACTIONABLE SUGGESTED FIX: {attempt['suggested_fix']}\n"
        
        sys_prompt += "\nADJUST YOUR MATH to strictly follow the SUGGESTED FIX above with your Chain of Thoughts for reasoning."

    user_prompt  = f"SPATIAL JSON:\n{state['spatial_json']}"
    
    propose_crop_agents = AGENT_REGISTRY["cropagent-propose-bbox"]
    prediction: SuggestedCropBox = await cortex_call_llm(
        model_configs=propose_crop_agents,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        images_b64=state['test_page_images_b64'],
        response_model=SuggestedCropBox
    )

    box = (max(0, prediction.left_x0), max(0, prediction.top_y0), min(p_w, prediction.right_x1), min(p_h, prediction.bottom_y1))

    logger.debug(f"   -> VLM Chain of Thought: {prediction.scratchpad}")
    logger.debug(f"   -> VLM Reasoning: {prediction.reasoning}")
        
    logger.info(f"🎯 Proposed Bounding Box: {box}")

    if DEBUG_MODE:
        # VISUALIZE THE NEWLY PROPOSED BOX on all sample pages
        for p_num in state["sample_pages"]:
            draw_and_save_bbox(state["pdf_path"], p_num, box, iteration, state["sponsor_name"])

    return {
        "proposed_box": box, 
        "iteration_count": iteration,
        "current_proposal_reasoning": prediction.reasoning,
        "debug_propose_prompt": sys_prompt
    }

async def node_validate_crop(state: CropState) -> CropState:

    doc = fitz.open(state["pdf_path"])
    box = state["proposed_box"]

    logger.info(f"✂️ CropAgent: Validating box {box} across sample pages: {state['sample_pages']}")
    
    # 1. Render annotated images with bbox drawn + collect text from all pages
    annotated_images = []
    all_headers = []
    all_footers = []
    all_body_starts = []
    all_body_ends = []
    
    for p_num in state["sample_pages"]:
        if p_num - 1 >= len(doc):
            continue
        page = doc[p_num - 1]
        
        # Render page with red bounding box drawn
        annotated_img = render_page_with_bbox(state["pdf_path"], p_num, box)
        if annotated_img:
            annotated_images.append(annotated_img)
        
        # Extract text zones
        cut_header = page.get_text("text", clip=fitz.Rect(0, 0, page.rect.width, box[1])).strip()
        cut_footer = page.get_text("text", clip=fitz.Rect(0, box[3], page.rect.width, page.rect.height)).strip()
        kept_body = page.get_text("text", clip=fitz.Rect(box)).strip()
        
        if cut_header:
            all_headers.append(f"[Page {p_num}] {cut_header}")
        if cut_footer:
            all_footers.append(f"[Page {p_num}] {cut_footer}")
        if kept_body:
            all_body_starts.append(f"[Page {p_num}] {kept_body[:100]}")
            all_body_ends.append(f"[Page {p_num}] {kept_body[-100:]}")
    
    doc.close()
    
    # 2. Build combined prompt
    user_prompt = f"""PROPOSED CROP BOX: {box}
Sample Pages Analyzed: {state['sample_pages']}

--- TEXT CUT OUT AS HEADER (across all pages) ---
{chr(10).join(all_headers) if all_headers else '(none)'}

--- TEXT CUT OUT AS FOOTER (across all pages) ---
{chr(10).join(all_footers) if all_footers else '(none)'}

--- FIRST 100 CHARS OF BODY (to detect header text leaking into body) ---
{chr(10).join(all_body_starts) if all_body_starts else '(none)'}

--- LAST 100 CHARS OF BODY (to detect footer text leaking into body) ---
{chr(10).join(all_body_ends) if all_body_ends else '(none)'}

Look at the {len(annotated_images)} annotated images. The RED rectangle shows the proposed crop area that KEEPS the body text.
Compare header/footer text across pages — if the same text appears on multiple pages, it is likely a true header/footer.
Check the body edges: if the first 100 chars of body contain header-like text (e.g. document title, protocol number) that repeats across pages, the top crop is too high.
Similarly, if the last 100 chars of body contain footer-like text (e.g. page numbers, confidentiality notices) that repeats, the bottom crop is too low."""
    
    sys_prompt = """You are a crop validation expert. You receive annotated page images with a RED bounding box showing the proposed crop area, plus extracted text from the header zone, footer zone, and body edges.
Validate the crop by comparing all pages together. Think step-by-step:
1. Are the header/footer texts truly headers/footers (repeating across pages)?
2. Did the crop accidentally cut valid body text into the header/footer zones?
3. Did header/footer text leak INTO the body (check the first/last 100 chars)?
4. LEFT/RIGHT MARGIN CHECK: Look at the RED bounding box in the images. Does it have enough horizontal breathing space on the left and right sides? The box should NOT tightly hug the text edges — there must be visible whitespace padding (at least 10-15 points) on both the left and right sides. If the box is too tight horizontally and risks truncating text, return is_valid=False with a suggested_fix to widen the left_x0 and/or right_x1.
Return is_valid=False if any issue is found."""

    crop_validator_agents = AGENT_REGISTRY["cropagent-validate-bbox"]
    
    # 3. Single multi-image LLM call
    result: CropValidation = await cortex_call_llm(
        model_configs=crop_validator_agents,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        images_b64=annotated_images,
        response_model=CropValidation
    )

    # RICH DEBUG LOGGING
    logger.debug(f"   -> Validator Scratchpad: {result.scratchpad}")
    
    if not result.is_valid:
        logger.warning(f"❌ Validation FAILED across sample pages!")
        logger.debug(f"   -> Reason: {result.reasoning}")
        logger.debug(f"   -> Suggested Fix for next loop: {result.suggested_fix}")
    
        # SAVE THE RICH VALIDATION DATA
        failed_attempt = {
            "box": box, 
            "propose_reasoning": state.get("current_proposal_reasoning", ""),
            "val_reasoning": result.reasoning,
            "suggested_fix": result.suggested_fix
        }
        updated_history = state.get("attempt_history", []) + [failed_attempt]
        
        return {
            "is_valid": False,
            "validation_reasoning": result.reasoning, 
            "iteration_count": state["iteration_count"] + 1,
            "attempt_history": updated_history,
            "debug_validate_prompt": user_prompt
        }
    
    # Passed validation on all pages!
    logger.info(f"🏆 CROP VALIDATED SUCCESSFULLY on all sample pages!")
    logger.debug(f"   -> Saving box {box} to memory for sponsor '{state['sponsor_name']}'.")
            
    save_memory(state["sponsor_name"], box)
    return {"is_valid": True, "validation_reasoning": "Passed all pages.", "debug_validate_prompt": user_prompt}

# --- Routing ---
def route_crop(state: CropState):
    if state["is_valid"]: return "end"
    
    # We allow more retries because the first few might just be testing historical boxes
    total_attempts_allowed = len(state["historical_boxes"]) + 3 
    
    if state["iteration_count"] >= total_attempts_allowed:
        logger.error("🚨 CropAgent: Maximum retries reached. Giving up on dynamic cropping.")
        return "end" # Give up, master will fallback to full page
    return "retry"

crop_workflow = StateGraph(CropState)
crop_workflow.add_node("extract", node_extract_and_load_memory)
crop_workflow.add_node("propose", node_propose_crop)
crop_workflow.add_node("validate", node_validate_crop)

crop_workflow.set_entry_point("extract")
crop_workflow.add_edge("extract", "propose")
crop_workflow.add_edge("propose", "validate")
crop_workflow.add_conditional_edges("validate", route_crop, {"end": END, "retry": "propose"})

graph = crop_workflow.compile()