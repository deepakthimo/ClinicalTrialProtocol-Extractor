from typing import TypedDict, List, Dict, Any, Literal, Annotated
from pydantic import BaseModel, Field
import operator

class SynopsisMapping(BaseModel):
    """Page numbers for the Synopsis section extracted from the TOC."""
    synopsis_pages: List[int] = Field(description="List of exact integer page numbers for the Synopsis.")

class MappedSection(BaseModel):
    """A single section mapped from the PDF table of contents."""
    normalized_title: str = Field(description="The standard name of the section from the required list.")
    actual_pdf_title: str = Field(description="The exact section title exactly as it appears in the PDF TOC.")
    start_page: int = Field(description="The starting page number of the section.")
    end_page: int = Field(description="The ending page number (the page before the next major section starts).")

class SectionMapping(BaseModel):
    """Collection of mapped sections from the PDF."""
    sections: List[MappedSection] = Field(description="List of successfully mapped required sections.")

class PageState(TypedDict):
    page_num: int
    mode: Literal["synopsis", "section"]
    system_prompt: str
    target_section: str
    image_b64: str
    pymupdf_text: str
    previous_context: str

    # Internal working memory
    vlm_raw_text: str
    final_markdown: str
    validation_feedback: List[str]
    iteration_count: int
    validation_status: str
    error_source: str

    # Best-of-N tracking
    attempt_history: Annotated[List[dict], operator.add]
    page_failed: bool

class MasterState(TypedDict):
    job_id: str
    pdf_url: str
    sponsor_name: str
    required_normalized_sections: List[str]
    
    # Internal Prep State
    pdf_path: str
    crop_box: tuple # (x0, y0, x1, y1)
    toc_text: str
    toc_images_b64: list[str]
    pdf_status: str
    
    # Generated Mapping
    synopsis_pages: List[int]
    section_pages: List[MappedSection]
    
    # Abort Tracking
    abort_node: str   # which node triggered the early END (e.g. "validate", "check_synopsis")
    abort_reason: str # LLM-generated reason explaining why the pipeline was aborted

    # Extraction State
    raw_pages: Dict[int, Dict[str, str]]
    cleaned_synopsis: str
    cleaned_sections: Dict[str, str]
    final_dataset: Dict[str, dict]

class CropState(TypedDict):
    pdf_path: str
    sponsor_name: str
    sample_pages: List[int] # e.g., [10, 15, 22]
    
    # Multimodal Payload for LLM
    spatial_json: str
    test_page_images_b64: List[str]
    
    # Memory & Validation
    historical_boxes: List[tuple]
    proposed_box: tuple
    is_valid: bool
    iteration_count: int
    memory_index_tested: int

    attempt_history: List[dict]

    # For reasoning and debugging 
    validation_reasoning: str
    current_proposal_reasoning: str
    debug_propose_prompt: str


class ValidationResult(BaseModel):
    """Result of validating extracted markdown against the source image."""
    status: Literal["PASS", "FAIL"] = Field(description="PASS if the markdown accurately reflects the rules, FAIL if there are errors.")
    error_source: Literal["LAYOUT_EXTRACTION", "RULE_FOLLOWING", "NONE"] = Field(
        description="If FAIL, indicate if the error is due to missing/jumbled text layout from the image (LAYOUT_EXTRACTION), or failure to follow Markdown/formatting rules (RULE_FOLLOWING). If PASS, use NONE."
    )
    discrepancies: List[str] = Field(description="If FAIL, list the specific errors. If PASS, leave empty.")

class SynopsisQualityCheck(BaseModel):
    """Quality check for extracted synopsis content."""
    status: Literal["PASS", "FAIL"] = Field(description="PASS if the body of the synospis contain some level of information to process.")
    reason: str = Field(description="Explanation for status.")

class PDFQualityCheck(BaseModel):
    """Quality check for PDF text extraction readability."""
    status: Literal["PASS", "FAIL"] = Field(description="PASS if text is readable, FAIL if text is corrupted/fragmented with spaces (e.g. 'C l i n i c a l').")
    reason: str = Field(description="Explanation for status.")

class SuggestedCropBox(BaseModel):
    """Proposed bounding box coordinates for cropping page content."""
    scratchpad: str = Field(description="Chain of thought: Analyze the spatial data, previous mistakes, and do the math step-by-step before deciding coordinates.")
    top_y0: float = Field(description="The Y-coordinate just BELOW the header.")
    bottom_y1: float = Field(description="The Y-coordinate just ABOVE the footer.")
    left_x0: float = Field(description="The X-coordinate just to the RIGHT of any left-vertical text.")
    right_x1: float = Field(description="The X-coordinate just to the LEFT of any right-vertical text.")
    reasoning: str = Field(description="Explain why you chose these coordinates based on the text blocks.")

class CropValidation(BaseModel):
    """Validation result for a proposed crop bounding box."""
    scratchpad: str = Field(description="Analyze the text cut as headers/footers. Did we accidentally cut valid clinical sentences? Think step-by-step.")
    suggested_fix: str = Field(description="Actionable math advice (e.g., 'Move top_y0 UP by 20px', 'Move bottom_y1 DOWN').")
    is_valid: bool = Field(description="True if the header/footer was successfully removed but NO body text was lost.")
    reasoning: str = Field(description="Explanation of the validation.")

class FinalDatasetItem(BaseModel):
    """A single fine-tuning dataset item with instruction, input, and output."""
    instruction: str = Field(description="The instructional prompt for fine-tuning")
    input: str = Field(description="The context input (synopsis)")
    output: str = Field(description="The extracted section body")
