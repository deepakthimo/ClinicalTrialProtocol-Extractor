from typing import Dict, Any

def create_initial_state(job_id: str, pdf_url: str, sponsor_name: str) -> Dict[str, Any]:
    """
    Factory function to create the initial state for the master graph.
    Ensures that both the CLI (main.py) and the API (api.py) use the exact same
    base structure, preventing state mismatch bugs.
    """
    return {
        "job_id": job_id,
        "pdf_url": pdf_url,
        "sponsor_name": sponsor_name,
        "required_normalized_sections": [
            'objectives and endpoints', 'introduction', 'study design',
            'study assessments and procedures', 'study population',
            'treatments', 'statistical considerations', 'investigational plan',
            'discontinuation of study treatment and participant discontinuation and withdrawal',
            'description of design and trial population',
            'supporting documentation and operational considerations',
            'statistical methods and determination of sample size'
        ],
        "pdf_path": "",
        "toc_text": "",
        "toc_images_b64": [],
        "pdf_status": "",
        "abort_node": "",
        "abort_reason": "",
        "crop_box": (0, 0, 0, 0),
        "synopsis_pages": [],
        "section_pages": [],
        "raw_pages": {},
        "cleaned_synopsis": "",
        "cleaned_sections": {},
        "final_dataset": {}
    }
