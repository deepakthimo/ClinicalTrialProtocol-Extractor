import time
import requests

BASE_URL = "http://localhost:8000"
POLL_INTERVAL_SECONDS = 300  # 5 minutes between status checks

TERMINAL_STATUSES = {"COMPLETED", "FAILED"}


def prompt_inputs() -> tuple[str, list[str]]:
    """Interactively ask the user for sponsor name and PDF URLs."""
    print("=" * 60)
    print("  Lilly PDF Extractor — Batch Submit")
    print("=" * 60)

    sponsor = input("\nSponsor name (e.g. 'Pfizer', 'Boehringer Ingelheim'): ").strip()
    if not sponsor:
        raise SystemExit("❌ Sponsor name cannot be empty.")

    print("\nEnter PDF URLs one per line.")
    print("Leave a blank line when done:\n")
    urls = []
    while True:
        url = input(f"  URL {len(urls) + 1}: ").strip()
        if not url:
            break
        if not url.startswith("http"):
            print("  ⚠️  Skipped — does not look like a valid URL.")
            continue
        urls.append(url)

    if not urls:
        raise SystemExit("❌ At least one PDF URL is required.")

    print(f"\n✅ Sponsor : {sponsor}")
    print(f"✅ PDFs    : {len(urls)}")
    for u in urls:
        print(f"   • {u}")
    print()
    return sponsor, urls


def submit(sponsor: str, pdf_urls: list[str]) -> list[dict]:
    """POST all URLs as a single batch and return the list of job dicts."""
    payload = {
        "requests": [
            {"pdf_url": url, "sponsor_name": sponsor}
            for url in pdf_urls
        ]
    }

    print(f"Submitting {len(pdf_urls)} PDF(s) for sponsor: '{sponsor}'")
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
    results = {}  # {job_id: {"status": str, "error_message": str | None}}

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
                    results[job_id] = {
                        "status": status,
                        "error_message": data.get("error_message"),
                    }
                    completed_this_round.append(job_id)
                    if status == "FAILED":
                        error_msg = data.get("error_message", "No error message provided")
                        print(f"  [{job_id[:8]}...]  ❌ REASON: {error_msg}")

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
    for job_id, info in results.items():
        status = info["status"]
        icon = "✅" if status == "COMPLETED" else "❌"
        print(f"  {icon}  {job_id}  →  {status}")
        if status == "FAILED" and info.get("error_message"):
            print(f"       Reason: {info['error_message']}")
    print()

    total = len(results)
    succeeded = sum(1 for info in results.values() if info["status"] == "COMPLETED")
    print(f"  {succeeded}/{total} jobs completed successfully.")


if __name__ == "__main__":
    sponsor, pdf_urls = prompt_inputs()
    jobs = submit(sponsor, pdf_urls)
    poll(jobs)
