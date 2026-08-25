"""Generate a synthetic insurance certificate PDF plus its matching gold JSON.

Only for smoke-testing the pipeline without a real document. The real POC run
uses data/input.pdf and data/gold.json supplied by hand.

    python tests/make_sample_pdf.py            # -> data/sample_input.pdf, data/sample_gold.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import ROOT, pymupdf, write_json  # noqa: E402

GOLD = {
    "document_type": "certificate_of_insurance",
    "policy_number": "GL-0042198-01",
    "insurer_name": "Northbridge Mutual Insurance Company",
    "insured_name": "Harbour Point Logistics Inc.",
    "insured_address": "1450 Dockside Road, Suite 300, Halifax, NS B3J 2K9",
    "broker_name": "Cormorant Risk Partners",
    "effective_date": "2025-04-01",
    "expiration_date": "2026-04-01",
    "issue_date": "2025-03-18",
    "currency": "CAD",
    "total_premium": 18450.0,
    "taxes_and_fees": 1476.0,
    "coverages": [
        {"coverage_type": "Commercial General Liability", "limit": 5000000.0,
         "deductible": 2500.0, "premium": 12300.0},
        {"coverage_type": "Non-Owned Automobile", "limit": 2000000.0,
         "deductible": 1000.0, "premium": 3150.0},
        {"coverage_type": "Tenants Legal Liability", "limit": 500000.0,
         "deductible": 1000.0, "premium": 3000.0},
    ],
    "notes": "Coverage is primary and non-contributory in favour of the certificate holder.",
}

LINES = [
    ("CERTIFICATE OF INSURANCE", 18, True),
    ("", 11, False),
    ("Northbridge Mutual Insurance Company", 12, True),
    ("Issued through: Cormorant Risk Partners", 11, False),
    ("Date issued: March 18, 2025", 11, False),
    ("", 11, False),
    ("Policy Number: GL-0042198-01", 12, True),
    ("Named Insured: Harbour Point Logistics Inc.", 11, False),
    ("Address: 1450 Dockside Road, Suite 300, Halifax, NS B3J 2K9", 11, False),
    ("", 11, False),
    ("Policy Period: April 1, 2025 to April 1, 2026 (12:01 a.m. local time)", 11, False),
    ("Currency: CAD", 11, False),
    ("", 11, False),
    ("SCHEDULE OF COVERAGES", 12, True),
    ("Coverage                          Limit        Deductible     Premium", 10, False),
    ("Commercial General Liability      5,000,000    2,500          12,300.00", 10, False),
    ("Non-Owned Automobile              2,000,000    1,000           3,150.00", 10, False),
    ("Tenants Legal Liability             500,000    1,000           3,000.00", 10, False),
    ("", 11, False),
    ("Total Premium: $18,450.00", 11, True),
    ("Taxes and Fees: $1,476.00", 11, False),
    ("", 11, False),
    ("Remarks: Coverage is primary and non-contributory in favour of the", 10, False),
    ("certificate holder.", 10, False),
]


def variant(index: int) -> dict:
    """A synthetic sibling of the sample document: same shape, different values."""
    gold = json.loads(json.dumps(GOLD))
    gold["policy_number"] = f"GL-00421{index:02d}-01"
    gold["insured_name"] = f"Harbour Point Logistics {index} Inc."
    gold["total_premium"] = 18450.0 + index * 100
    gold["taxes_and_fees"] = 1476.0 + index * 8
    for c in gold["coverages"]:
        c["premium"] = round(c["premium"] + index * 25, 2)
    return gold


def build_pdf(pdf_path: Path, gold: dict | None = None) -> None:
    fitz = pymupdf()

    lines = LINES
    if gold is not None:
        replace = {
            "Policy Number: GL-0042198-01": f"Policy Number: {gold['policy_number']}",
            "Named Insured: Harbour Point Logistics Inc.": f"Named Insured: {gold['insured_name']}",
            "Total Premium: $18,450.00": f"Total Premium: ${gold['total_premium']:,.2f}",
            "Taxes and Fees: $1,476.00": f"Taxes and Fees: ${gold['taxes_and_fees']:,.2f}",
        }
        lines = [(replace.get(t, t), size, bold) for t, size, bold in LINES]

    doc = fitz.open()
    page = doc.new_page()  # A4 default
    y = 70
    for text, size, bold in lines:
        if text:
            page.insert_text(
                (60, y), text, fontsize=size,
                fontname="hebo" if bold else "helv", color=(0, 0, 0),
            )
        y += size + 7
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(pdf_path)
    doc.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a synthetic sample document + gold JSON.")
    ap.add_argument("--pdf", default=str(ROOT / "data" / "sample_input.pdf"))
    ap.add_argument("--gold", default=str(ROOT / "data" / "sample_gold.json"))
    args = ap.parse_args()

    build_pdf(Path(args.pdf))
    write_json(Path(args.gold), GOLD)
    print(f"wrote {args.pdf}")
    print(f"wrote {args.gold}")
    print(json.dumps(GOLD, indent=2)[:200] + " ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
