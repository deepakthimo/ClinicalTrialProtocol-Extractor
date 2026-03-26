import fitz
import requests
import tempfile
import base64
import random
import re
from core.logger import setup_logger
logger = setup_logger()


def download_pdf(url: str, target_path: str = None) -> str:
    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        # Check if the content type is roughly a PDF (or missing, to be safe)
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            logger.warning(f"Warning: URL returned HTML instead of PDF. URL: {url}")
            
        if target_path:
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return target_path
        else:
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            for chunk in response.iter_content(chunk_size=8192):
                temp_pdf.write(chunk)
            temp_pdf.close()
            return temp_pdf.name
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        raise

def analyze_pdf_content(doc: fitz.Document):
    """Scans PDF for TOC, returns combined text and base64 images of those pages."""
    toc_text_list = []
    toc_image_list = []
    toc_found = False
    toc_finished = False
    
    toc_start_keywords = ["table of contents", "table of content"] 
    toc_end_keywords = ["references", "appendix", "appendices", "index"]

    for i in range(min(len(doc), 40)): # Only check first 40 pages
        if toc_finished: break
        
        page = doc[i]
        text = page.get_text()
        text_lower = text.lower()
        
        is_start_page = any(k in text_lower for k in toc_start_keywords)
        if is_start_page:
            logger.debug(f"TOC Search: Potential TOC start found on page {i+1} using keywords.")

        if not toc_found:
            if is_start_page:
                toc_found = True
                logger.debug(f"TOC Search: TOC confirmed started on page {i+1}")
                toc_text_list.append(text)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                toc_image_list.append(base64.b64encode(pix.tobytes("jpeg")).decode('utf-8'))
                
                # Check if end keyword is on same page
                if any(k in text_lower for k in toc_end_keywords) and len(text) > 500:
                    logger.debug(f"TOC Search: TOC end keyword found on the same page as start (Page {i+1})")
                    toc_finished = True
        else:
            logger.debug(f"TOC Search: Continuing TOC extraction on page {i+1}")
            toc_text_list.append(text)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            toc_image_list.append(base64.b64encode(pix.tobytes("jpeg")).decode('utf-8'))
            
            if any(k in text_lower for k in toc_end_keywords):
                logger.debug(f"TOC Search: TOC end keyword found on page {i+1}")
                toc_finished = True
            elif len(toc_text_list) > 8:
                logger.debug(f"TOC Search: TOC exceeded 8 pages limit. Ending extraction.")
                toc_finished = True

    # Fallback if TOC keywords completely fail
    if not toc_found:
        logger.warning("No TOC found using keywords. Falling back to Page 2/3.")
        fallback_page = doc[2] if len(doc) > 2 else doc[0]
        toc_text_list.append(fallback_page.get_text())
        pix = fallback_page.get_pixmap(matrix=fitz.Matrix(2, 2))
        toc_image_list.append(base64.b64encode(pix.tobytes("jpeg")).decode('utf-8'))

    return "\n".join(toc_text_list), toc_image_list

def calculate_dynamic_crop_box(doc: fitz.Document, target_pages) -> tuple:
    
    # Randomly select up to 4 pages from the targets to check repeating headers
    sample_pages = random.sample(list(target_pages), min(4, len(target_pages)))
    
    page_width, page_height = doc[0].rect.width, doc[0].rect.height
    highest_footer_y0, lowest_header_y1 = page_height * 0.92, page_height * 0.08  
    
    blocks_by_page = []
    for p_num in sample_pages:
        if p_num - 1 < len(doc): 
            blocks_by_page.append(doc[p_num - 1].get_text("blocks"))

    if len(blocks_by_page) >= 2:
        def clean_text(t): return re.sub(r'\d+', '', t).strip().lower()
        for b1 in blocks_by_page[0]:
            if b1[5] != 0 or len(clean_text(b1[4])) < 3: continue
            for b2 in blocks_by_page[1]:
                if clean_text(b2[4]) == clean_text(b1[4]) and abs(b1[1] - b2[1]) < 10:
                    if b1[3] < page_height * 0.3: lowest_header_y1 = max(lowest_header_y1, b1[3])
                    elif b1[1] > page_height * 0.7: highest_footer_y0 = min(highest_footer_y0, b1[1])
                    
    doc.close()
    crop_tuple = (0, lowest_header_y1 + 5, page_width, highest_footer_y0 - 5)
    logger.debug(f"Dynamic Crop: Calculated Box -> {crop_tuple} (Page W: {page_width}, H: {page_height})")
    return crop_tuple

def merge_page_text(prev_text: str, next_text: str) -> str:
    """
    Dynamically merges text from two consecutive pages based on punctuation, 
    capitalization, and markdown list formatting.
    """
    # 1. Handle empty strings safely
    if not prev_text: return next_text
    if not next_text: return prev_text

    # 2. Clean boundaries (remove trailing spaces from prev, leading from next)
    prev_text = prev_text.rstrip()
    next_text = next_text.lstrip()
    
    if not prev_text or not next_text:
        return prev_text + next_text

    # 3. Extract boundary lines and characters for evaluation
    last_line = prev_text.split('\n')[-1].strip()
    first_line = next_text.split('\n')[0].strip()
    
    last_char = prev_text[-1]
    first_char = next_text[0]
    
    # Define sentence-ending punctuation (added '!' and '?' just in case)
    terminators = ('.', '!', '?', ':')

    # Helper function to detect if a line is a bullet point (handles "-", "*", or "1.")
    def is_list_item(line):
        return bool(re.match(r'^(\-|\*|\+|\d+\.)\s', line))

    # ==========================================
    # RULE 1: Bullet Point Continuation
    # ==========================================
    if is_list_item(last_line) and is_list_item(first_line):
        return prev_text + "\n" + next_text

    # ==========================================
    # RULE 2: Incomplete Sentences Wrapping Pages
    # ==========================================
    has_terminator = prev_text.endswith(terminators)
    is_next_capitalized = first_char.isupper()

    # Case A: No full stop, no capital letter -> Mid-sentence break
    if not has_terminator and not is_next_capitalized:
        return prev_text + " " + next_text

    # Case B: No full stop, BUT next word is capitalized. 
    # (Edge case: Proper nouns crossing pages, e.g., "Conducted by \n Pfizer")
    if not has_terminator and is_next_capitalized:
        return prev_text + " " + next_text

    # Case C: Has full stop, BUT next word is lowercase.
    # (Edge case: Abbreviations wrapping pages, e.g., "e.g. \n taking aspirin")
    if has_terminator and not is_next_capitalized:
        return prev_text + " " + next_text

    # ==========================================
    # RULE 3: Perfect Paragraph Break
    # ==========================================
    # Has full stop AND next page starts with a capital letter
    if has_terminator and is_next_capitalized:
        return prev_text + "\n\n" + next_text

    # Fallback default
    return prev_text + "\n\n" + next_text

def clean_markdown_headers(text: str) -> str:
    """
    Finds markdown headers and removes the numbering between the hash and the title.
    Examples:
        '# 3. STUDY DESIGN' -> '# STUDY DESIGN'
        '## 3.1. Study Overview' -> '## Study Overview'
        '### 1.2.3.4 Inclusion' -> '### Inclusion'
    """
    # Regex Explanation:
    # ^(#+)          : Matches the beginning of a line and captures 1 or more '#'
    # \s*            : Matches optional spaces
    # \d+(?:\.\d+)*\.? : Matches numbers like '3', '3.1', '3.1.2.', '10.'
    # \s+            : Matches at least one space after the numbers
    # (.*)           : Captures the rest of the line (the actual title)
    pattern = re.compile(r'^(#+)\s*\d+(?:\.\d+)*\.?\s+(.*)', re.MULTILINE)
    
    # Replace with just the hashes, a space, and the text
    return pattern.sub(r'\1 \2', text)

def force_normalized_top_header_section(text: str, norm_title: str) -> str:
    """
    Finds the first Level 1 markdown header ('# ...') and replaces it entirely 
    with the normalized title. 
    """
    # Regex Explanation:
    # ^#\s+.*$ : Matches exactly one '#' followed by spaces and any text to the end of the line.
    pattern = re.compile(r'^#\s+.*$', re.MULTILINE)
    
    # We uppercase the normalized title so it looks like a standard main header
    normalized_header = f"# {norm_title.upper()}"
    
    # Replace only the first occurrence (count=1)
    new_text = pattern.sub(normalized_header, text, count=1)
    
    # Fallback: If for some reason the VLM didn't generate a '#' header at the top,
    # we prepend it manually so the output always strictly starts with the normalized header.
    if not re.search(r'^#\s+', text, re.MULTILINE):
        new_text = f"{normalized_header}\n\n{new_text}"
        
    return new_text

def remove_top_level_header_synopsis(text: str) -> str:
    """
    Finds the first Level 1 markdown header ('# ...') and removes it entirely,
    leaving only the body of the text.
    """
    # Regex Explanation:
    # ^#\s+    : Matches exactly one '#' followed by spaces at the start of a line.
    # .*       : Matches the rest of the title text.
    # \n*      : Matches any newlines immediately following the title to prevent empty gaps.
    pattern = re.compile(r'^#\s+.*\n*', re.MULTILINE)
    
    # Replace the first occurrence with an empty string
    return pattern.sub('', text, count=1).strip()

