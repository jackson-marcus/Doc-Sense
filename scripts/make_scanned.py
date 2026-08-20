"""Generate scan-degraded PDFs with known ground truth for OCR benchmarking.

Renders synthetic business documents (reportlab), rasterizes them, applies
noise/rotation (PIL), and saves image-only PDFs plus the ground-truth text.
This gives an honest, measurable OCR accuracy story without any dataset
download.

Usage:
    uv run python scripts/make_scanned.py [--count 3]
"""

from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docsense.settings import get_config, resolve_path

DOCS = [
    (
        "invoice-alpha",
        [
            "INVOICE #A-1001",
            "Bill to: Northwind Traders",
            "Item: Industrial pump unit, quantity 4",
            "Unit price: 2,150.00 USD",
            "Total due: 8,600.00 USD",
            "Payment terms: net 30 days",
        ],
    ),
    (
        "contract-beta",
        [
            "SERVICE AGREEMENT",
            "This agreement is between Contoso Ltd and Fabrikam Inc.",
            "The service period begins on 1 March 2026.",
            "Monthly fee: 12,000.00 USD payable in advance.",
            "Either party may terminate with 60 days written notice.",
        ],
    ),
    (
        "report-gamma",
        [
            "QUARTERLY OPERATIONS REPORT",
            "Production output rose 14 percent quarter over quarter.",
            "Downtime was reduced to 3.2 hours per week.",
            "Two safety incidents were recorded, down from five.",
            "Headcount at quarter end: 412 employees.",
        ],
    ),
]


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.rotate(rng.uniform(-1.5, 1.5), expand=False, fillcolor="white")
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
    # Speckle noise
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(200, 600)):
        x, y = rng.randint(0, img.width - 1), rng.randint(0, img.height - 1)
        draw.point((x, y), fill="black" if rng.random() < 0.5 else "gray")
    return img


def render_page(lines: list[str]) -> Image.Image:
    img = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(img)
    y = 150
    for line in lines:
        draw.text((120, y), line, fill="black")
        y += 90
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=len(DOCS))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    raw_dir = resolve_path(get_config()["ingestion"]["raw_dir"])
    truth_dir = raw_dir.parent / "ocr_ground_truth"
    raw_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    for name, lines in DOCS[: args.count]:
        img = degrade(render_page(lines), rng)
        pdf_path = raw_dir / f"{name}.pdf"
        buf = io.BytesIO()
        img.save(buf, "PDF")
        pdf_path.write_bytes(buf.getvalue())
        (truth_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"{name}: scan-degraded PDF -> {pdf_path}")

    print("\nGround truth in", truth_dir)


if __name__ == "__main__":
    main()
