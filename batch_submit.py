import time
import requests

# ==============================================================================
# CONFIGURE THESE BEFORE RUNNING
# ==============================================================================
BASE_URL = "http://localhost:8000"
POLL_INTERVAL_SECONDS = 300  # 5 minutes between status checks; adjust as needed based on expected processing time

SPONSOR_NAME = "Gilend Sciences"

PDF_URLS = [
    "https://cdn.clinicaltrials.gov/large-docs/95/NCT05770895/Prot_000.pdf",
    "https://cdn.clinicaltrials.gov/large-docs/44/NCT05765344/Prot_000.pdf",
    "https://cdn.clinicaltrials.gov/large-docs/00/NCT05760300/Prot_000.pdf",
]
# ==============================================================================

TERMINAL_STATUSES = {"completed", "failed"}


def submit() -> list[dict]:
    """POST all URLs as a single batch and return the list of job dicts."""
    payload = {
        "requests": [
            {"pdf_url": url, "sponsor_name": SPONSOR_NAME}
            for url in PDF_URLS
        ]
    }

    print(f"Submitting {len(PDF_URLS)} PDF(s) for sponsor: '{SPONSOR_NAME}'")
    print("-" * 60)

    response = requests.post(f"{BASE_URL}/api/v1/extract/batch", json=payload)
    response.raise_for_status()

    data = response.json()
    print(f"Batch ID : {data['batch_id']}")
    print(f"Jobs     : {len(data['jobs'])}\n")

    for job in data["jobs"]:
        print(f"  Job ID     : {job['job_id']}")
        print(f"  Status URL : {BASE_URL}{job['status_url']}")
        print()

    return data["jobs"]


def poll(jobs: list[dict]):
    """Poll all jobs until every one reaches a terminal state."""
    # Track pending jobs as {job_id: status_url}
    pending = {job["job_id"]: job["status_url"] for job in jobs}
    results = {}  # {job_id: final_status}

    print("=" * 60)
    print("Polling for results...")
    print("=" * 60)

    while pending:
        time.sleep(POLL_INTERVAL_SECONDS)

        completed_this_round = []
        for job_id, status_path in pending.items():
            try:
                resp = requests.get(f"{BASE_URL}{status_path}")
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "unknown")

                print(f"  [{job_id[:8]}...]  status: {status.upper()}")

                if status in TERMINAL_STATUSES:
                    results[job_id] = status
                    completed_this_round.append(job_id)

            except requests.RequestException as e:
                print(f"  [{job_id[:8]}...]  ERROR polling status: {e}")

        for job_id in completed_this_round:
            del pending[job_id]

        if pending:
            print(f"\n  {len(pending)} job(s) still running. Next check in {POLL_INTERVAL_SECONDS}s...\n")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for job_id, status in results.items():
        icon = "✅" if status == "completed" else "❌"
        print(f"  {icon}  {job_id}  →  {status.upper()}")
    print()

    total = len(results)
    succeeded = sum(1 for s in results.values() if s == "completed")
    print(f"  {succeeded}/{total} jobs completed successfully.")


if __name__ == "__main__":
    jobs = submit()
    poll(jobs)
