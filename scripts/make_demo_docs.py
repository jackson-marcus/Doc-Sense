"""Generate small digital demo PDFs with known facts (for demos and eval).

Usage:
    uv run python scripts/make_demo_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docsense.settings import get_config, resolve_path

DOCS = {
    "acme-annual-2025": [
        [
            "ACME Corporation — Annual Report 2025",
            "Total revenue was 12.5 million dollars in fiscal year 2025,",
            "an increase of 18 percent over the prior year.",
            "Gross margin improved to 61 percent.",
        ],
        [
            "Risk factors: supply chain concentration in a single region",
            "remains the primary operational risk.",
            "The company employs 340 people worldwide.",
            "A share buyback program of 2 million dollars was approved in March.",
        ],
    ],
    "globex-annual-2025": [
        [
            "Globex Industries — Annual Report 2025",
            "Globex reported a net loss of 3 million dollars for the year,",
            "driven by one-time restructuring charges.",
            "Cash reserves stood at 21 million dollars at year end.",
        ],
        [
            "During the year the company opened a new research facility in Berlin",
            "focused on battery storage technology.",
            "Employee headcount grew to 5,000 across all offices.",
            "The dividend was suspended until profitability returns.",
        ],
    ],
}


def main() -> None:
    raw_dir = resolve_path(get_config()["ingestion"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, pages in DOCS.items():
        path = raw_dir / f"{name}.pdf"
        c = canvas.Canvas(str(path), pagesize=letter)
        for lines in pages:
            y = 720
            for line in lines:
                c.drawString(72, y, line)
                y -= 22
            c.showPage()
        c.save()
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
