"""Download a small corpus of real SEC EDGAR 10-K filings as PDFs.

SEC serves filings as HTML; we fetch the primary document for a few well-known
companies and convert to PDF-like text corpus. To keep this dependency-light we
save the raw HTML and a plain-text extraction; the RAG pipeline also accepts
PDFs dropped into data/raw_pdfs manually.

SEC fair-access policy requires a descriptive User-Agent with contact email.

Usage:
    uv run python scripts/fetch_edgar.py [--email you@example.com]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docsense.settings import get_config, resolve_path

# (company, CIK) — CIKs are stable public identifiers.
COMPANIES = [
    ("apple", "0000320193"),
    ("microsoft", "0000789019"),
    ("tesla", "0001318605"),
]

BASE = "https://data.sec.gov"


def latest_10k_url(cik: str, client: httpx.Client) -> str | None:
    r = client.get(f"{BASE}/submissions/CIK{cik}.json")
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    for form, accession, doc in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"], strict=False
    ):
        if form == "10-K":
            acc = accession.replace("-", "")
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", default="docsense-demo@example.com")
    args = parser.parse_args()

    out_dir = resolve_path(get_config()["ingestion"]["raw_dir"]).parent / "edgar_html"
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": f"docsense research demo {args.email}"}

    with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
        for name, cik in COMPANIES:
            url = latest_10k_url(cik, client)
            if not url:
                print(f"{name}: no 10-K found")
                continue
            r = client.get(url)
            r.raise_for_status()
            path = out_dir / f"{name}-10k.html"
            path.write_bytes(r.content)
            print(f"{name}: saved {len(r.content):,} bytes -> {path}")
            time.sleep(0.5)  # be polite to SEC servers

    print("\nConvert to PDF or drop PDFs into data/raw_pdfs, then run: make ingest")


if __name__ == "__main__":
    main()
