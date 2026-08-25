"""Independent audit of the golden JSON corpus.

The generator validates its own output, which is circular: it checks a record
against a page it rendered from that same record. This re-derives every check
from the files on disk instead, so a bug in the generator cannot hide from it.

    python src/verify_gold.py                # audit data/
    python src/verify_gold.py --ocr          # also OCR each PDF and check grounding
    python src/verify_gold.py --json report.json

Checks, per document:
  parse         the gold is valid JSON and non-empty
  pages         document_metadata.total_pages equals the PDF's real page count
  arithmetic    per-vehicle premium columns sum to their totals, and the parts
                sum to the term amount
  dates         parse, and are ordered (transaction <= effective, expiry = +1yr)
  formats       VIN length, ZIP digits, money formatting, state abbreviations
  emptiness     no null/blank where the rest of the corpus carries a value

and across the corpus:
  schema        every document has an identical key skeleton
  duplicates    no two documents are the same
  variability   which fields actually differ between documents

With --ocr it additionally reads each PDF through the project's OCR stage and
checks that every scalar in the gold appears on the page (soundness) and that
the page holds nothing the gold omits (completeness).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    discover_documents, load_config, pymupdf, read_json, resolve, setup_logging, write_json,
)

log = setup_logging("verify_gold")

MONEY = re.compile(r"^\$-?[\d,]+(?:\.\d{2})?$")
ZIP_RE = re.compile(r"^\d{5}$")
STATE_RE = re.compile(r"^[A-Z]{2}$")
DATE_FMT = "%m/%d/%Y"

# Fields that are a classification of the document rather than text printed on it.
# They cannot be grounded against the page, so they are counted separately - listed
# in the report, never silently excused.
DERIVED_FIELDS = {"document_type"}


# --------------------------------------------------------------------------- helpers


# When a list holds objects, one of these identifies the row far better than its
# position does: "the Comprehensive row" is comparable across documents, "row 3"
# is not, because different documents order and populate rows differently.
IDENTITY_KEYS = ("description", "coverage", "nature_of_interest", "type", "label", "name")


def leaves(obj, prefix="", by_identity=False):
    """Every scalar in the document, as (path, value).

    by_identity keys list items by their identifying field rather than their
    index, so the same logical row can be compared across documents.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaves(value, f"{prefix}.{key}" if prefix else key, by_identity)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            token = str(i)
            if by_identity and isinstance(value, dict):
                for key in IDENTITY_KEYS:
                    if isinstance(value.get(key), str) and value[key].strip():
                        token = value[key].strip()
                        break
            yield from leaves(value, f"{prefix}[{token}]", by_identity)
    else:
        yield prefix, obj


def skeleton(obj, prefix=""):
    """Key shape, ignoring list lengths, so documents can be compared."""
    out = []
    if isinstance(obj, dict):
        for key in sorted(obj):
            out += skeleton(obj[key], f"{prefix}.{key}")
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], (dict, list)):
            out += skeleton(obj[0], prefix + "[]")
        else:
            out.append(prefix + "[]")
    else:
        out.append(prefix)
    return out


def money(value):
    """'$1,234.00' -> 1234.0, else None."""
    if not isinstance(value, str):
        return None
    text = value.strip().replace("$", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_date(value):
    try:
        return datetime.strptime(str(value).strip(), DATE_FMT).date()
    except (ValueError, TypeError):
        return None


def dig(obj, *path):
    """Safe nested lookup; returns None if any step is missing."""
    for step in path:
        if isinstance(step, int):
            if not isinstance(obj, list) or len(obj) <= step:
                return None
            obj = obj[step]
        else:
            if not isinstance(obj, dict) or step not in obj:
                return None
            obj = obj[step]
    return obj


# --------------------------------------------------------------------------- checks


def check_pages(gold, pdf):
    """total_pages must match what the shipped PDF actually contains."""
    claimed = dig(gold, "document_metadata", "total_pages")
    if claimed is None:
        return ["document_metadata.total_pages is missing"]
    try:
        fitz = pymupdf()
    except ImportError:
        return []
    with fitz.open(pdf) as doc:
        actual = doc.page_count
    if claimed != actual:
        return [f"total_pages says {claimed} but the PDF has {actual} page(s)"]
    return []


def check_arithmetic(gold):
    """Premium columns must sum to their totals, and the parts to the term amount."""
    problems = []
    vc = dig(gold, "states", 0, "locations", 0, "vehicle_coverages")
    if not isinstance(vc, dict):
        return ["states[0].locations[0].vehicle_coverages is missing"]
    rows = vc.get("rows") or []
    totals = vc.get("totals") or []
    columns = vc.get("vehicle_columns") or []
    if len(totals) != len(columns):
        problems.append(f"{len(columns)} vehicle column(s) but {len(totals)} total(s)")

    for i in range(min(len(totals), len(columns))):
        summed = 0.0
        for row in rows:
            cell = (row.get("premiums") or [None] * len(totals))[i] if isinstance(row, dict) else None
            amount = money(cell)
            if amount is not None:
                summed += amount
        declared = money(totals[i])
        if declared is None:
            problems.append(f"vehicle column {i + 1} total {totals[i]!r} is not money")
        elif abs(summed - declared) > 0.01:
            problems.append(
                f"vehicle column {i + 1}: premiums sum to {summed:,.2f} "
                f"but the total says {declared:,.2f}")

    grand = sum(money(t) or 0.0 for t in totals) + (money(dig(gold, "line_of_business", "total")) or 0.0)
    term = money(dig(gold, "transaction", "term_amount"))
    if term is None:
        problems.append("transaction.term_amount is not money")
    elif abs(grand - term) > 0.01:
        problems.append(f"parts sum to {grand:,.2f} but term_amount says {term:,.2f}")
    return problems


def check_dates(gold):
    problems = []
    txn = parse_date(dig(gold, "transaction", "transaction_date"))
    eff = parse_date(dig(gold, "transaction", "effective_date"))
    exp = parse_date(dig(gold, "transaction", "expiration_date"))
    printed = parse_date(dig(gold, "document_metadata", "printed_on"))

    for name, value in (("transaction_date", txn), ("effective_date", eff),
                        ("expiration_date", exp), ("printed_on", printed)):
        if value is None:
            problems.append(f"{name} does not parse as {DATE_FMT}")
    if txn and eff and txn > eff:
        problems.append(f"transaction_date {txn} is after effective_date {eff}")
    if eff and exp:
        try:
            expected = date(eff.year + 1, eff.month, eff.day)
        except ValueError:                                   # 29 Feb
            expected = None
        if expected and exp != expected:
            problems.append(f"expiration_date {exp} is not one year after effective_date {eff}")
    if txn and printed and printed < txn:
        problems.append(f"printed_on {printed} precedes transaction_date {txn}")
    return problems


def check_formats(gold):
    problems = []
    for i, vehicle in enumerate(gold.get("vehicles") or [], start=1):
        vin = vehicle.get("vin")
        if not isinstance(vin, str) or len(vin) != 17:
            problems.append(f"vehicles[{i}].vin is {vin!r} ({len(vin) if isinstance(vin, str) else '?'} chars, expected 17)")
    for path, value in leaves(gold):
        if path.endswith(".zip") and not (isinstance(value, str) and ZIP_RE.match(value)):
            problems.append(f"{path} = {value!r} is not a 5-digit ZIP")
        if path.endswith(".state") and isinstance(value, str) and len(value) == 2 \
                and not STATE_RE.match(value):
            problems.append(f"{path} = {value!r} is not a 2-letter state code")
        if isinstance(value, str) and value.strip().startswith("$") and not MONEY.match(value.strip()):
            problems.append(f"{path} = {value!r} is malformed money")
    return problems


def check_emptiness(gold, populated, others):
    """Blank here where EVERY other document fills the same logical field.

    A blank cell is usually legitimate - a coverage with a limit but no
    deductible, say - so this only fires when this document is the sole one
    missing a value, which is the shape a genuine extraction slip takes.
    """
    problems = []
    if others < 1:
        return problems
    for path, value in leaves(gold, by_identity=True):
        blank = value is None or (isinstance(value, str) and not value.strip())
        if blank and populated.get(path, 0) == others:
            problems.append(f"{path} is blank, but all {others} other document(s) fill it")
    return problems


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit the golden JSON corpus.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None, help="overrides paths.data_dir")
    ap.add_argument("--ocr", action="store_true",
                    help="also OCR each PDF and check the gold against the page text")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = resolve(args.data_dir) if args.data_dir else resolve(cfg["paths"]["data_dir"])
    documents, unpaired_pdfs, unpaired_golds = discover_documents(data_dir)
    if not documents:
        log.error("no PDF+gold pairs under %s", data_dir)
        return 2

    log.info("auditing %d document(s) in %s", len(documents), data_dir)
    for pdf in unpaired_pdfs:
        log.warning("PDF with no gold, not audited: %s", pdf.name)
    for gold in unpaired_golds:
        log.warning("gold with no PDF: %s", gold.name)

    golds, failed = {}, {}
    for doc in documents:
        try:
            golds[doc.doc_id] = read_json(doc.gold)
        except Exception as exc:
            failed[doc.doc_id] = f"gold does not parse: {exc}"

    # How many documents fill each field: used to spot a value missing in just one.
    populated = defaultdict(int)
    for gold in golds.values():
        for path, value in leaves(gold, by_identity=True):
            if value is not None and not (isinstance(value, str) and not value.strip()):
                populated[path] += 1

    results = {}
    for doc in documents:
        if doc.doc_id in failed:
            results[doc.doc_id] = [failed[doc.doc_id]]
            continue
        gold = golds[doc.doc_id]
        problems = []
        problems += check_pages(gold, doc.pdf)
        problems += check_arithmetic(gold)
        problems += check_dates(gold)
        problems += check_formats(gold)
        problems += check_emptiness(gold, populated, len(golds) - 1)
        results[doc.doc_id] = problems

    # corpus-wide
    corpus_problems = []
    skeletons = {doc_id: skeleton(g) for doc_id, g in golds.items()}
    if skeletons:
        reference_id = sorted(skeletons)[0]
        reference = skeletons[reference_id]
        for doc_id, shape in sorted(skeletons.items()):
            if shape != reference:
                extra = sorted(set(shape) - set(reference))[:5]
                missing = sorted(set(reference) - set(shape))[:5]
                corpus_problems.append(
                    f"{doc_id}: key shape differs from {reference_id}"
                    + (f"; extra {extra}" if extra else "")
                    + (f"; missing {missing}" if missing else ""))

    seen = {}
    for doc_id, gold in sorted(golds.items()):
        digest = json.dumps(gold, sort_keys=True)
        if digest in seen:
            corpus_problems.append(f"{doc_id} is byte-identical to {seen[digest]}")
        seen[digest] = doc_id

    varying = 0
    if len(golds) > 1:
        values = defaultdict(set)
        for gold in golds.values():
            for path, value in leaves(gold):
                values[re.sub(r"\[\d+\]", "[]", path)].add(repr(value))
        varying = sum(1 for v in values.values() if len(v) > 1)

    # optional grounding against the page text
    grounding = {}
    if args.ocr:
        grounding = run_grounding(cfg, documents, golds)
        if not grounding:
            log.error("--ocr was requested but no OCR output was found. Run "
                      "'python src/run_ocr.py' first (scanned PDFs need MinerU). "
                      "Grounding was NOT checked.")
        if not grounding:
            log.error("--ocr was requested but no OCR output was found for any document. "
                      "Run 'python src/run_ocr.py' first (scanned PDFs need MinerU). "
                      "Grounding was NOT checked.")

    # A value that is not on its page is a per-document defect, so it must show up
    # in that document's row - not only in the closing verdict.
    for doc_id, g in grounding.items():
        for item in g.get("ungrounded", []):
            results.setdefault(doc_id, []).append(
                f"{item['field']} = {item['value']!r} does not appear on the page")

    # ---------------------------------------------------------------- report
    width = max(len(d) for d in results) + 2
    print(f"\n{'DOCUMENT'.ljust(width)} {'RESULT':>10}  DETAIL")
    print("-" * (width + 40))
    clean = 0
    for doc_id in sorted(results):
        problems = results[doc_id]
        if not problems:
            clean += 1
            extra = ""
            if doc_id in grounding:
                g = grounding[doc_id]
                printed = g["scalars"] - len(g.get("derived_not_printed") or [])
                extra = (f"grounded {g['grounded']}/{printed} printed values via "
                         f"{g['source']}, page word coverage {g['coverage']:.0%}")
            print(f"{doc_id.ljust(width)} {'OK':>10}  {extra}")
        else:
            print(f"{doc_id.ljust(width)} {'PROBLEM':>10}  {problems[0]}")
            for extra in problems[1:]:
                print(f"{' '.ljust(width)} {'':>10}  {extra}")
    print("-" * (width + 40))
    print(f"{clean}/{len(results)} document(s) clean")
    if len(golds) > 1:
        print(f"schema: {'identical across the corpus' if not corpus_problems else 'INCONSISTENT'}")
        print(f"fields that vary between documents: {varying}")
    for problem in corpus_problems:
        print(f"  - {problem}")

    report = {
        "data_dir": str(data_dir),
        "documents": len(documents),
        "clean": clean,
        "per_document": results,
        "corpus_problems": corpus_problems,
        "varying_fields": varying,
        "grounding": grounding,
    }
    if args.json:
        write_json(resolve(args.json), report)
        print(f"written: {resolve(args.json)}")

    total_problems = sum(len(v) for v in results.values()) + len(corpus_problems)
    if total_problems:
        print(f"\nVERDICT: {total_problems} problem(s) found")
        return 1
    if grounding:
        derived = sorted({d["field"] for g in grounding.values()
                          for d in (g.get("derived_not_printed") or [])})
        if derived:
            print(f"derived fields, not printed on the page (not a defect): {', '.join(derived)}")
        ungrounded = sum(g["ungrounded_total"] for g in grounding.values())
        if ungrounded:
            print(f"\nVERDICT: {ungrounded} value(s) in the gold do not appear on their page")
            return 1
        print("\nVERDICT: consistent, and every value was found on its page")
        return 0
    if args.ocr:
        # Never report "grounded" when grounding did not actually run.
        print("\nVERDICT: structural checks passed, but grounding was NOT verified "
              "(no OCR output - see the error above)")
        return 1
    print("\nVERDICT: structural checks passed. Re-run with --ocr to also verify that "
          "every value actually appears on its page.")
    return 0


def pdf_text_layer(pdf):
    """The PDF's own embedded text, if it has any. Empty for a scanned page."""
    try:
        fitz = pymupdf()
    except ImportError:
        return ""
    with fitz.open(pdf) as doc:
        return "\n".join(page.get_text("text") for page in doc).strip()


def page_text(cfg, doc):
    """Text of this document's pages, and where it came from.

    Prefers the OCR stage's output, which is what the model will actually read.
    Falls back to the PDF's embedded text layer - present when the generator ran
    with --no-scan - so grounding can be verified without MinerU.
    """
    from common import ocr_dir_for, require_pages
    from prompting import build_ocr_text

    try:
        return build_ocr_text(require_pages(ocr_dir_for(cfg, doc.doc_id))), "ocr"
    except FileNotFoundError:
        pass
    text = pdf_text_layer(doc.pdf)
    if text:
        return text, "pdf text layer"
    return "", "none"


def run_grounding(cfg, documents, golds):
    """Check the gold against what is actually on the page."""
    from compare import coverage_report

    out = {}
    for doc in documents:
        if doc.doc_id not in golds:
            continue
        text, source = page_text(cfg, doc)
        if not text:
            log.warning("[%s] no page text: run src/run_ocr.py, or generate with "
                        "--no-scan so the PDF keeps a text layer", doc.doc_id)
            continue
        gold = golds[doc.doc_id]
        flat = " ".join(text.split()).lower()
        scalars = [(p, v) for p, v in leaves(gold)
                   if v is not None and str(v).strip() and not isinstance(v, bool)]
        ungrounded, derived = [], []
        for path, value in scalars:
            needle = " ".join(str(value).split()).lower()
            if needle and needle not in flat:
                target = derived if path in DERIVED_FIELDS else ungrounded
                target.append({"field": path, "value": value})
        cov = coverage_report(text, gold)
        out[doc.doc_id] = {
            "source": source,
            "scalars": len(scalars),
            # Derived fields are not printed, so they are neither grounded nor a defect.
            "grounded": len(scalars) - len(ungrounded) - len(derived),
            "ungrounded": ungrounded[:25],
            "ungrounded_total": len(ungrounded),
            "derived_not_printed": derived,
            "coverage": cov.get("word_coverage") or 0.0,
            "number_coverage": cov.get("number_coverage"),
            "missing_numbers": cov.get("missing_numbers", [])[:20],
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
