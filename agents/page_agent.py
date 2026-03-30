import re
import logging

from core.state import PageState, ValidationResult
from langgraph.graph import StateGraph, END
from agents.cortex_langchain import cortex_call_llm
from agents.agent_registry import AGENT_REGISTRY

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

def _build_cumulative_feedback(attempts: list[dict], max_history: int = 3) -> str:
    """Build a cumulative error history block from recent attempts."""
    recent = attempts[-max_history:]
    if not recent:
        return ""

    lines = ["\nCRITICAL - HISTORY OF ALL PREVIOUS FAILURES (do NOT repeat ANY of these):"]
    for i, attempt in enumerate(recent, 1):
        discs = attempt.get("discrepancies", [])
        if discs:
            source = attempt.get("error_source", "UNKNOWN")
            lines.append(f"\nAttempt {i} ({source}):")
            lines.extend(f"  - {d}" for d in discs)
    lines.append("\nYou MUST fix ALL listed errors while NOT regressing on previously fixed ones.")
    return "\n".join(lines)


async def node_extract_vlm(state: PageState) -> PageState:
    iteration = state.get("iteration_count", 0)

    logger.info(f"📄 PageAgent: [Page {state['page_num']}] Step 1: VLM Visual Extraction (Attempt {iteration + 1})...")

    sys_prompt = "Extract all text from this image exactly as you see it. Maintain paragraph breaks. Do not format, just transcribe."
    user_prompt = ""

    # Cumulative feedback from ALL prior attempts (not just the last one)
    if iteration > 0 and state.get("error_source") == "LAYOUT_EXTRACTION":
        attempts = state.get("attempt_history", [])
        sys_prompt += _build_cumulative_feedback(attempts)
        sys_prompt += "\n\nPay closer attention to the visual reading order and do not skip text."

    extractor_agents = AGENT_REGISTRY["pageagent-extractor"]

    raw_text = await cortex_call_llm(
        model_configs=extractor_agents,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        images_b64=state['image_b64']
    )
    if isinstance(raw_text, list):
        text_parts = [block.get("text", "") for block in raw_text if isinstance(block, dict) and "text" in block]
        raw_text = "\n".join(text_parts)
    elif not isinstance(raw_text, str):
        raw_text = str(raw_text)

    snippet = raw_text[:400].replace("\n", " ") + "..."
    logger.debug(f"VLM Raw Extraction Snippet: {snippet}")

    return {"vlm_raw_text": raw_text}


async def node_reconcile_llm(state: PageState) -> PageState:
    iteration = state.get("iteration_count", 0)
    logger.info(f"📄 PageAgent: [Page {state['page_num']}] Step 2: LLM Reconciliation (Attempt {iteration + 1})...")

    sys_prompt = f"""You are an expert Clinical AI. Your job is to construct perfect Markdown for the section: "{state['target_section']}".

You have two sources of truth:
1. PyMuPDF Text (Perfect spelling, bad layout) - Stictly use PyMuPDF text for spellings.
2. Vision Model Text (Good layout, might have typos) - Strictly follow Vision model text for layout.

CRITICAL RULES:
1. DO NOT extract tables, figures, footnotes, or headers/footers.
2. Format headings based on their numbering (e.g., '1.' = #, '1.2.' = ##, '1.2.3.' = ###, '1.2.3.4' = ####).
3. Output ONLY the text belonging to "{state['target_section']}". If a new section starts, STOP extracting.
4. Output verbatim text. Do not summarize. Do not wrap in ```markdown tags.
5. If there is no text for the target section on this page, output nothing.
6. Between each paragraph there should be paragraph break (\\n\\n) and there shouldnt be any line break with single (\\n) in the text."""

    # Cumulative feedback from ALL prior attempts (not just the last one)
    feedback_block = ""
    if iteration > 0:
        attempts = state.get("attempt_history", [])
        if attempts:
            feedback_block = _build_cumulative_feedback(attempts)

            logger.debug("Injecting Cumulative Validator Feedback from all previous attempts:")
            for i, attempt in enumerate(attempts[-3:], 1):
                for fb in attempt.get("discrepancies", []):
                    logger.debug(f"      Attempt {i} - {fb}")

    context_block = f"<previous_page_context>\n{state['previous_context']}\n</previous_page_context>\n" if state['previous_context'] else ""

    if state['previous_context']:
        logger.debug(f"Injecting Previous Context (tail): {state['previous_context'][-150:].replace(chr(10), ' ')}...")

    user_prompt = f"""
    {feedback_block}
    {context_block}
    <pymupdf_source>\n{state['pymupdf_text']}\n</pymupdf_source>
    <vision_source>\n{state['vlm_raw_text']}\n</vision_source>
    
    Construct the final cleaned Markdown strictly adhering to the SYSTEM rules.
    """

    reconcile_agents = AGENT_REGISTRY["pageagent-reconciler"]

    final_md = await cortex_call_llm(
        model_configs=reconcile_agents,
        system_prompt=sys_prompt,
        user_prompt=user_prompt
    )

    md_snippet = final_md[:400].replace("\n", " \\n ") + "..." if final_md else "[NO TEXT EXTRACTED]"
    logger.debug(f"LLM Generated Markdown Snippet: {md_snippet}")

    return {"final_markdown": final_md.strip()}


async def node_validate_vlm(state: PageState) -> PageState:
    logger.info(f"📄 PageAgent: [Page {state['page_num']}] Step 3: VLM Validation...")

    val_rules = ""
    if state["mode"] == "section":
        val_rules = f"""- Target Section: "{state['target_section']}".
- Tables, figures, and page numbers MUST BE EXCLUDED. (Do not fail if a table is missing, it is supposed to be).
- Did the markdown miss any actual text from the section?
- Did the markdown hallucinate bullet points?"""
    else:
        val_rules = """- The Synopsis table MUST be flattened into text.
- No markdown tables allowed.
- Did the markdown miss any actual text from the section?
- Did the markdown hallucinate bullet points?"""

    sys_prompt = f"""You are a strict QA Validator. Compare the generated Markdown text to the original IMAGE.
    
    RULES TO CHECK AGAINST:
    {val_rules}
    
    DIAGNOSING ERRORS:
    If the text fails, you MUST diagnose the root cause:
    - Choose "LAYOUT_EXTRACTION" if the text is fundamentally missing large chunks, hallucinated content, or the reading order is completely mangled compared to the image. 
    - Choose "RULE_FOLLOWING" if the text is mostly there, but it failed to follow the specific formatting rules (e.g., included tables, wrong headers, bad paragraph breaks).
    
    Return PASS if it successfully followed rules. Return FAIL with discrepancies and the error source if it failed."""

    user_prompt = f"<generated_markdown>\n{state['final_markdown']}\n</generated_markdown>\nValidate markdown against image."

    validate_agents = AGENT_REGISTRY["pageagent-validator"]

    result: ValidationResult = await cortex_call_llm(
        model_configs=validate_agents,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        images_b64=state['image_b64'],
        response_model=ValidationResult
    )

    if result.status == "FAIL":
        logger.warning(f"❌ VALIDATION FAILED on Page {state['page_num']}!")
        logger.warning(f"   -> Root Cause Diagnosed: {result.error_source}")
    else:
        logger.info(f"✅ PASSED VALIDATION on Page {state['page_num']}!")

    # Save this attempt for Best-of-N selection
    current_attempt = {
        "markdown": state["final_markdown"],
        "discrepancies": result.discrepancies,
        "error_source": result.error_source,
    }

    return {
        "validation_feedback": result.discrepancies,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "validation_status": result.status,
        "error_source": result.error_source if result.status == "FAIL" else "NONE",
        "attempt_history": [current_attempt],   # Annotated[..., operator.add] appends
    }


# ── Best-of-N: LLM Judge picks the best from all failed attempts ──
async def node_select_best(state: PageState) -> PageState:
    attempts = state.get("attempt_history", [])
    logger.info(f"🏆 PageAgent: [Page {state['page_num']}] Selecting best from {len(attempts)} failed attempts...")

    # Trivial case — skip the LLM call
    if len(attempts) <= 1:
        chosen = attempts[0] if attempts else {"markdown": state.get("final_markdown", "")}
        logger.warning(f"🏆 Only {len(attempts)} attempt(s) available — using it as-is.")
        return {"final_markdown": chosen["markdown"], "page_failed": True}

    # Build the comparison prompt (text-only, no image needed)
    candidates_block = ""
    for i, attempt in enumerate(attempts, 1):
        discs = attempt.get("discrepancies", [])
        disc_text = "\n".join(f"  - {d}" for d in discs) if discs else "  (none listed)"
        candidates_block += (
            f"\n--- CANDIDATE {i} ---\n"
            f"Error type: {attempt.get('error_source', 'UNKNOWN')}\n"
            f"Discrepancies ({len(discs)}):\n{disc_text}\n"
            f"Markdown:\n{attempt['markdown']}\n"
        )

    sys_prompt = (
        "You are a Quality Selection Agent for clinical document extraction.\n"
        "You will see multiple extraction attempts for the SAME page, each with "
        "validator-identified discrepancies.\n\n"
        "Pick the BEST candidate — the one with the fewest and least severe errors.\n"
        "Prefer RULE_FOLLOWING errors over LAYOUT_EXTRACTION errors (text is at least present).\n\n"
        "Reply with ONLY the candidate number (e.g. 1 or 2). Nothing else."
    )
    user_prompt = f"Select the best candidate:\n{candidates_block}"

    # Re-uses reconciler agents (text-only call, no new agent config needed)
    selector_agents = AGENT_REGISTRY["pageagent-fallback-best-effort-selector"]

    response = await cortex_call_llm(
        model_configs=selector_agents,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
    )

    # Parse the number from the LLM response
    match = re.search(r"\d+", response.strip())
    if match:
        idx = int(match.group()) - 1
        if 0 <= idx < len(attempts):
            logger.info(f"🏆 LLM Judge selected Candidate {idx + 1} for Page {state['page_num']}.")
            return {"final_markdown": attempts[idx]["markdown"], "page_failed": True}

    # Fallback: deterministic pick — fewest discrepancies, prefer RULE_FOLLOWING over LAYOUT
    logger.warning(f"🏆 LLM Judge returned unparseable response. Falling back to fewest-discrepancy heuristic.")
    best = min(
        attempts,
        key=lambda a: (
            len(a.get("discrepancies", [])),
            0 if a.get("error_source") == "RULE_FOLLOWING" else 1,
        ),
    )
    return {"final_markdown": best["markdown"], "page_failed": True}


# ── Adaptive retry + route to select_best on exhaustion ──
def should_continue(state: PageState):
    if state.get("validation_status") == "PASS":
        logger.info(f"✅ PageAgent: Successfully Validated Page {state['page_num']}. CONTENT PASS!!!")
        return "end"

    error_source = state.get("error_source", "RULE_FOLLOWING")
    iteration = state["iteration_count"]

    # Adaptive early-stop: same error type 3x in a row -> not going to self-correct
    attempts = state.get("attempt_history", [])
    if len(attempts) >= 3:
        last_3 = [a["error_source"] for a in attempts[-3:]]
        if all(s == error_source for s in last_3):
            logger.warning(
                f"🛑 PageAgent: Same error type '{error_source}' repeated 3x consecutively on Page {state['page_num']}. "
                "Early stopping -> selecting best attempt."
            )
            return "select_best"

    # Hard cap: 5 total attempts
    if iteration >= 5:
        logger.error(f"🚨 PageAgent: Max retries (5) reached for Page {state['page_num']}. Selecting best attempt.")
        return "select_best"

    # Route retry to the right actor
    if error_source == "LAYOUT_EXTRACTION":
        logger.info(f"🔄 PageAgent: Layout flaw detected. Retrying VLM Extraction (Actor 1) for Page {state['page_num']}...")
        return "retry_extract"
    else:
        logger.info(f"🔄 PageAgent: Formatting rule flaw detected. Retrying LLM Reconciler (Actor 2) for Page {state['page_num']}...")
        return "retry_reconcile"


# ── GRAPH WIRING ──
page_workflow = StateGraph(PageState)

page_workflow.add_node("extract", node_extract_vlm)
page_workflow.add_node("reconcile", node_reconcile_llm)
page_workflow.add_node("page_validate", node_validate_vlm)
page_workflow.add_node("select_best", node_select_best)

page_workflow.set_entry_point("extract")
page_workflow.add_edge("extract", "reconcile")
page_workflow.add_edge("reconcile", "page_validate")
page_workflow.add_edge("select_best", END)

page_workflow.add_conditional_edges(
    "page_validate",
    should_continue,
    {
        "end": END,
        "retry_extract": "extract",
        "retry_reconcile": "reconcile",
        "select_best": "select_best",
    },
)

graph = page_workflow.compile()