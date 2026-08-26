"""Stage 6 - score the base model against a fine-tuned version, on the gold JSON.

Scoring is field level over the flattened gold JSON, with light normalization so
formatting differences (currency symbols, thousands separators, date styles,
casing, whitespace) do not count as extraction errors.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    adapter_dir_for,
    cfg_path,
    is_version,
    load_config,
    model_version,
    read_json,
    results_dir_for,
    setup_logging,
    validate_against_schema,
    write_json,
)

log = setup_logging("compare")

NULLISH = {"", "null", "none", "n/a", "na", "-", "--", "not stated", "not provided"}
DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    "%d.%m.%Y", "%Y%m%d",
]
MONEY_RE = re.compile(r"^[^\d\-+]*([-+]?\d[\d,\s']*(?:\.\d+)?)\s*[A-Za-z$€£¥]{0,3}$")
DATEISH_KEY = re.compile(r"(date|dated|expiry|expiration|effective|issued)", re.IGNORECASE)
# Identifiers stay strings: "007" must not be scored equal to 7.
IDISH_KEY = re.compile(r"(number|certificate|identifier|_id$|_no$|code)", re.IGNORECASE)


# --------------------------------------------------------------------------- flatten


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten to dot/bracket leaf paths. Empty containers become explicit leaves."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj:
            flat[prefix or "<root>"] = {}
            return flat
        for key, value in obj.items():
            flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        flat[f"{prefix}[]#count"] = len(obj)
        for i, value in enumerate(obj):
            flat.update(flatten(value, f"{prefix}[{i}]"))
    else:
        flat[prefix or "<root>"] = obj
    return flat


# --------------------------------------------------------------------------- normalize


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, str):
        m = MONEY_RE.match(value.strip())
        if m:
            digits = re.sub(r"[,\s']", "", m.group(1))
            try:
                return round(float(digits), 2)
            except ValueError:
                return None
    return None


def as_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().replace(",", ", ").replace("  ", " ")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "")).date().isoformat()
    except ValueError:
        return None


def normalize(path: str, value: Any, keep_identifier: bool = True) -> Any:
    """Map a raw leaf to its comparable form.

    keep_identifier stops identifier-named fields being read as numbers, so
    "007" is not scored equal to 7. score() turns it off when the gold value is
    itself a number (e.g. a genuinely numeric "number_of_units" field).
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.casefold() in NULLISH:
            return None
        if DATEISH_KEY.search(path):
            iso = as_date(stripped)
            if iso:
                return iso
        if keep_identifier and IDISH_KEY.search(path):
            return re.sub(r"\s+", " ", stripped).strip(" .,;:").casefold()
        number = as_number(stripped)
        if number is not None:
            return number
        return re.sub(r"\s+", " ", stripped).strip(" .,;:").casefold()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return value


# --------------------------------------------------------------------------- scoring


def score(gold: dict, candidate: Any) -> dict:
    gold_flat = flatten(gold)
    cand_flat = flatten(candidate) if isinstance(candidate, (dict, list)) else {}

    fields = []
    matches = 0
    for path, gold_value in sorted(gold_flat.items()):
        present = path in cand_flat
        raw = cand_flat.get(path)
        keep_id = isinstance(gold_value, str)
        g_norm = normalize(path, gold_value, keep_id)
        c_norm = normalize(path, raw, keep_id) if present else "<missing>"
        hit = present and g_norm == c_norm
        matches += hit
        fields.append({
            "field": path,
            "gold": gold_value,
            "got": raw if present else None,
            "present": present,
            "match": bool(hit),
        })

    extra = sorted(set(cand_flat) - set(gold_flat))
    total = len(gold_flat)
    lists = {
        path[: -len("[]#count")]: {"gold": value, "got": cand_flat.get(path)}
        for path, value in gold_flat.items()
        if path.endswith("[]#count")
    }
    return {
        "total_fields": total,
        "matched_fields": matches,
        "match_rate": round(matches / total, 4) if total else 0.0,
        "missing_fields": [f["field"] for f in fields if not f["present"]],
        "extra_fields": extra,
        "list_counts": lists,
        "fields": fields,
    }


# Words too common to prove anything about coverage.
STOPWORDS = frozenset("""the a an and or of to in for on at by with from as is are was were
be been this that these those any all not no if then than which who whom whose you your we
our it its will shall may can must do does did have has had""".split())

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
NUMBER_RE = re.compile(r"\d[\d,./\-]*")


def _norm_number(token: str) -> str:
    return re.sub(r"[,\s$]", "", token).rstrip(".")


def content_tokens(text: str) -> tuple[set[str], set[str]]:
    """(words, numbers) worth checking for. Numbers carry most of the meaning here."""
    words = {w.lower() for w in WORD_RE.findall(text)} - STOPWORDS
    numbers = {n for n in (_norm_number(t) for t in NUMBER_RE.findall(text)) if len(n) >= 2}
    return words, numbers


def collect_values(obj: Any) -> str:
    """Every value in the extracted object, flattened to one searchable blob."""
    parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                parts.append(str(key))
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif node is not None:
            parts.append(str(node))

    walk(obj)
    return " ".join(parts)


def coverage_report(ocr_text: str, extracted: Any) -> dict:
    """How much of what OCR read off the pages actually reached the output.

    This is the completeness check: it is measured against the DOCUMENT, not
    against the gold, so it catches content the gold and the model both ignore.

    Caveats, and they matter: OCR noise produces tokens nothing will ever match,
    so 100% is not reachable, and presence is not correctness - a value in the
    wrong field still counts as covered. Read the base -> v1 delta, not the
    absolute number.
    """
    if extracted is None or not ocr_text.strip():
        return {}
    ocr_words, ocr_numbers = content_tokens(ocr_text)
    out_words, out_numbers = content_tokens(collect_values(extracted))

    missing_numbers = sorted(ocr_numbers - out_numbers)
    missing_words = sorted(ocr_words - out_words)
    return {
        "ocr_words": len(ocr_words),
        "ocr_numbers": len(ocr_numbers),
        "word_coverage": round(len(ocr_words & out_words) / len(ocr_words), 4) if ocr_words else None,
        "number_coverage": round(len(ocr_numbers & out_numbers) / len(ocr_numbers), 4)
        if ocr_numbers else None,
        "missing_numbers": missing_numbers[:60],
        "missing_numbers_total": len(missing_numbers),
        "missing_words_sample": missing_words[:60],
        "missing_words_total": len(missing_words),
    }


def field_variability(golds: list[dict]) -> dict[str, bool]:
    """Which fields actually differ between documents in the corpus.

    A large part of a form package is boilerplate - legal notices, disclosures,
    the printed question text - identical in every document of the type. A model
    learns those by rote, which inflates an overall match rate without any gain
    in extraction skill. Splitting the score on this tells you which happened.

    Needs at least two gold files; returns {} otherwise.
    """
    if len(golds) < 2:
        return {}
    flats = [flatten(g) for g in golds]
    paths: set[str] = set()
    for flat in flats:
        paths.update(flat)
    variable = {}
    for path in paths:
        seen = [repr(flat.get(path, "<absent>")) for flat in flats]
        variable[path] = len(set(seen)) > 1
    return variable


def split_by_variability(fields: list[dict], variable: dict[str, bool]) -> dict:
    """Match rates for the fields that vary across the corpus vs those that do not."""
    if not variable:
        return {}
    buckets = {"variable": [0, 0], "static": [0, 0]}
    for field in fields:
        if field["field"].endswith("#count"):
            continue
        bucket = "variable" if variable.get(field["field"], True) else "static"
        buckets[bucket][1] += 1
        buckets[bucket][0] += bool(field["match"])
    out = {}
    for name, (matched, total) in buckets.items():
        out[name] = {
            "matched": matched,
            "total": total,
            "match_rate": round(matched / total, 4) if total else None,
        }
    return out


def empty_score(gold: dict, reason: str) -> dict:
    result = score(gold, None)
    result["match_rate"] = 0.0
    result["matched_fields"] = 0
    for field in result["fields"]:
        field["match"] = False
    result["invalid_reason"] = reason
    return result


def load_candidate(results_dir: Path, name: str) -> tuple[Any | None, str, dict]:
    """Load a model's output plus the inference metadata that describes it.

    infer.py writes {name}_output.json even when the raw output was only a
    recoverable prefix, so strict JSON validity comes from the meta file, not
    from the mere existence of a parseable file.
    """
    meta_file = results_dir / "meta.json"
    meta = {}
    if meta_file.exists():
        try:
            meta = read_json(meta_file)
        except Exception:
            meta = {}
    path = results_dir / "output.json"
    if not path.exists():
        raw = results_dir / "raw.txt"
        note = "raw output exists but did not parse as JSON" if raw.exists() else "no output file"
        return None, f"{path.name} missing: {note}", meta
    try:
        return read_json(path), "ok", meta
    except Exception as exc:
        return None, f"{path.name} did not parse: {exc}", meta


# --------------------------------------------------------------------------- report


def section_of(path: str) -> str:
    """Top-level section a flattened field belongs to."""
    return path.split(".")[0].split("[")[0]


def section_breakdown(base: dict, v1: dict) -> list[dict]:
    """Per-top-level-section match rates, ordered by where v1 gains the most."""
    v1_by_field = {f["field"]: f for f in v1["fields"]}
    sections: dict[str, dict] = {}
    for field in base["fields"]:
        name = field["field"]
        if name.endswith("#count"):
            continue
        row = sections.setdefault(section_of(name), {"section": section_of(name), "fields": 0,
                                                     "base": 0, "trained": 0})
        row["fields"] += 1
        row["base"] += bool(field["match"])
        row["trained"] += bool(v1_by_field.get(name, {}).get("match"))
    out = []
    for row in sections.values():
        row["base_rate"] = round(row["base"] / row["fields"], 4)
        row["trained_rate"] = round(row["trained"] / row["fields"], 4)
        row["delta"] = round(row["trained_rate"] - row["base_rate"], 4)
        out.append(row)
    return sorted(out, key=lambda r: (-r["delta"], r["section"]))


def print_sections(rows: list[dict]) -> None:
    width = min(max((len(r["section"]) for r in rows), default=12), 44)
    header = f"{'SECTION'.ljust(width)}  {'FIELDS':>6}  {'BASE':>6}  {'TRAINED':>7}  {'DELTA':>7}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['section'][:width].ljust(width)}  {row['fields']:>6}  "
              f"{row['base_rate']:>6.0%}  {row['trained_rate']:>6.0%}  {row['delta']:>+7.0%}")
    print("-" * len(header))


def print_table(base: dict, v1: dict, limit: int, only_changed: bool) -> None:
    """Per-field detail. With 600+ fields, default to the rows where v1 differs from base."""
    v1_by_field = {f["field"]: f for f in v1["fields"]}
    rows = [f for f in base["fields"] if not f["field"].endswith("#count")]
    if only_changed:
        rows = [f for f in rows
                if bool(f["match"]) != bool(v1_by_field.get(f["field"], {}).get("match"))]
        caption = "FIELDS WHERE BASE AND V1 DISAGREE"
    else:
        caption = "ALL FIELDS"
    shown = rows[:limit]

    width = min(max((len(f["field"]) for f in shown), default=12), 52)
    header = f"{'FIELD'.ljust(width)}  {'BASE':^6}  {'V1':^6}  GOLD"
    print(f"\n{caption} ({len(rows)} of {len(base['fields'])})")
    print(header)
    print("-" * (len(header) + 20))
    for field in shown:
        path = field["field"]
        b = "  ok  " if field["match"] else " MISS "
        v = "  ok  " if v1_by_field.get(path, {}).get("match") else " MISS "
        name = path if len(path) <= width else ".." + path[-(width - 2):]
        value = str(field["gold"])
        if len(value) > 46:
            value = value[:43] + "..."
        print(f"{name.ljust(width)}  {b}  {v}  {value}")
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows; full detail in comparison.json")
    print("-" * (len(header) + 20))
    b_rate = f"{base['match_rate']:.0%}"
    v_rate = f"{v1['match_rate']:.0%}"
    print(f"{'MATCH RATE (all fields)'.ljust(width)}  {b_rate:^6}  {v_rate:^6}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score the base model against a fine-tuned version on the gold JSON.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--quiet", action="store_true", help="suppress the tables")
    ap.add_argument("--max-rows", type=int, default=40, help="cap on printed field rows")
    ap.add_argument("--all-fields", action="store_true",
                    help="print every field, not only those where base and v1 disagree")
    ap.add_argument("--doc", default=None, help="doc_id to score (default: the test document)")
    ap.add_argument("--version", default=None,
                    help="fine-tuned version to score against base, e.g. v1 / v2 "
                         "(default: model.version in config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset_meta_file = cfg_path(cfg, "dataset").parent / "dataset_meta.json"
    dataset_meta = read_json(dataset_meta_file) if dataset_meta_file.exists() else {}

    doc_id = args.doc or dataset_meta.get("test_document")
    if not doc_id:
        log.error("no test document recorded; run src/build_dataset.py or pass --doc")
        return 2
    doc_entry = next((d for d in dataset_meta.get("documents", [])
                      if d.get("doc_id") == doc_id), {})
    held_out = dataset_meta.get("test_held_out")
    log.info("scoring document %s (%s), corpus of %s document(s), %s training example(s)",
             doc_id,
             "held out of training" if held_out else "included in training",
             dataset_meta.get("corpus_size", "?"), dataset_meta.get("training_examples", "?"))

    gold_path = Path(doc_entry["gold"]) if doc_entry.get("gold") else None
    if gold_path is None or not gold_path.exists():
        from common import discover_documents

        documents, _, _ = discover_documents(cfg_path(cfg, "data_dir"))
        match = next((d for d in documents if d.doc_id == doc_id), None)
        if match is None:
            log.error("no gold JSON found for document %r", doc_id)
            return 2
        gold_path = match.gold
    gold = read_json(gold_path)
    version = args.version or model_version(cfg)
    if not is_version(version):
        log.error("--version must look like v1 / v2, got %r", version)
        return 2
    schema_file = cfg_path(cfg, "json_schema")
    log.info("comparing base vs %s on %s", version, doc_id)

    summary = {}
    scores = {}
    candidates: dict[str, Any] = {}
    models = ("base", version)
    for name in models:
        candidate, note, meta = load_candidate(results_dir_for(cfg, name, doc_id), name)
        candidates[name] = candidate
        loaded = candidate is not None
        repaired = bool(meta.get("repaired_from_truncation"))
        # Spec criterion 2 is strict: a repaired prefix is not valid JSON.
        valid = loaded and not repaired
        if meta and "json_valid_strict" in meta:
            valid = bool(meta["json_valid_strict"])
        schema_ok, schema_errors = (validate_against_schema(candidate, schema_file)
                                    if loaded else (False, [note]))
        scores[name] = score(gold, candidate) if loaded else empty_score(gold, note)
        summary[name] = {
            "json_valid": valid,
            "scored": loaded,
            "repaired_from_truncation": repaired,
            "hit_max_new_tokens": bool(meta.get("hit_max_new_tokens")),
            "note": note,
            "schema_ok": schema_ok,
            "schema_errors": schema_errors[:10],
            "conversation_fingerprint": meta.get("conversation_fingerprint"),
            "generated_tokens": meta.get("generated_tokens"),
            "confidence": meta.get("confidence") or {},
            "match_rate": scores[name]["match_rate"],
            "matched_fields": scores[name]["matched_fields"],
            "total_fields": scores[name]["total_fields"],
        }
        log.info("%-4s json_valid=%s%s schema_ok=%s match=%d/%d (%.1f%%)",
                 name, valid, " (scored from a repaired prefix)" if repaired else "",
                 schema_ok, scores[name]["matched_fields"],
                 scores[name]["total_fields"], scores[name]["match_rate"] * 100)

    base_rate = scores["base"]["match_rate"]
    v1_rate = scores[version]["match_rate"]
    verdict = "PASS" if v1_rate > base_rate else "INVESTIGATE"

    reasons = []
    if verdict == "PASS":
        reasons.append(f"{version} match rate {v1_rate:.1%} exceeds base {base_rate:.1%}")
        if v1_rate < 1.0:
            reasons.append(f"{version} does not fully reproduce the gold JSON; "
                           "more epochs would tighten the fit")
    else:
        if not summary[version]["scored"]:
            reasons.append(f"{version} produced nothing scoreable")
        elif abs(v1_rate - base_rate) < 1e-9:
            reasons.append(
                f"identical match rates: check that training loss fell, that merge.py ran, "
                f"and that infer.py --model {version} loaded the merged weights "
                f"(compare weight_signature in each model's meta.json)"
            )
        else:
            reasons.append(f"{version} scored below base: check prompt/modality drift "
                           "and the learning rate")

    for name in models:
        if summary[name]["repaired_from_truncation"]:
            capped = " after hitting max_new_tokens" if summary[name]["hit_max_new_tokens"] else ""
            reasons.append(
                f"WARNING: {name} output was NOT valid JSON{capped}; it was scored from the "
                f"recoverable prefix, so its match rate is a floor, not a measurement "
                f"(spec criterion 2 fails for {name})"
            )
    for name in models:
        meta = results_dir_for(cfg, name, doc_id) / "meta.json"
        if meta.exists():
            summary[name]["weight_signature"] = read_json(meta).get("weight_signature")
    sig_base = summary["base"].get("weight_signature")
    sig_v1 = summary[version].get("weight_signature")
    if sig_base is not None and sig_base == sig_v1:
        reasons.append(f"WARNING: base and {version} have identical weight signatures - "
                       f"{version} may be the base model")

    train_report = adapter_dir_for(cfg, version) / "train_report.json"
    training = read_json(train_report) if train_report.exists() else None
    if training and not training.get("loss_decreased"):
        reasons.append("WARNING: training loss did not decrease (see outputs/adapter/train_report.json)")

    # A crashed inference stage leaves the previous run's *_output.json in place.
    # Both results must carry the fingerprint of the dataset currently on disk.
    current_fp = doc_entry.get("fingerprint") or dataset_meta.get("test_fingerprint")
    if current_fp:
        for name in models:
            result_fp = summary[name].get("conversation_fingerprint")
            if result_fp and result_fp != current_fp:
                reasons.append(
                    f"WARNING: {name} result is STALE - produced from conversation "
                    f"{result_fp}, the dataset on disk is {current_fp}. Re-run "
                    f"infer.py --model {name}."
                )

    # Completeness: what fraction of what OCR read off the pages reached the output.
    # Measured against the document, so it catches content the gold omits too.
    coverage = {}
    try:
        from common import ocr_dir_for, require_pages
        from prompting import build_ocr_text

        ocr_text = build_ocr_text(require_pages(ocr_dir_for(cfg, doc_id)))
        for name in models:
            report_ = coverage_report(ocr_text, candidates.get(name))
            if report_:
                coverage[name] = report_
        gold_cov = coverage_report(ocr_text, gold)
        if gold_cov:
            coverage["gold"] = gold_cov
    except Exception as exc:
        log.warning("could not measure OCR coverage: %s", exc)

    if coverage.get("base") and coverage.get(version):
        log.info("OCR number coverage: base %.1f%% -> v1 %.1f%% (gold itself %.1f%%)",
                 (coverage["base"]["number_coverage"] or 0) * 100,
                 (coverage[version]["number_coverage"] or 0) * 100,
                 (coverage.get("gold", {}).get("number_coverage") or 0) * 100)
        if coverage[version]["missing_numbers_total"]:
            log.info("v1 left %d number(s) from the OCR out of its output, e.g. %s",
                     coverage[version]["missing_numbers_total"],
                     ", ".join(coverage[version]["missing_numbers"][:8]))

    # Split the score: fields that vary across the corpus are the real extraction
    # work; fields identical in every document are boilerplate the model can learn
    # by rote. A gain that is all boilerplate is not an extraction gain.
    variability = {}
    try:
        from common import discover_documents

        corpus_docs, _, _ = discover_documents(cfg_path(cfg, "data_dir"))
        golds = []
        for entry in corpus_docs:
            try:
                golds.append(read_json(entry.gold))
            except Exception:
                continue
        variability = field_variability(golds)
    except Exception as exc:
        log.warning("could not assess field variability across the corpus: %s", exc)

    split = {}
    if variability:
        n_var = sum(1 for v in variability.values() if v)
        log.info("corpus field variability: %d of %d field(s) differ between documents",
                 n_var, len(variability))
        for name in models:
            split[name] = split_by_variability(scores[name]["fields"], variability)
        v_base = split["base"].get("variable", {}).get("match_rate")
        v_v1 = split[version].get("variable", {}).get("match_rate")
        if v_base is not None and v_v1 is not None:
            reasons.append(
                f"on the fields that actually vary between documents (the extraction work), "
                f"base {v_base:.1%} -> v1 {v_v1:.1%}"
            )
    else:
        log.info("field variability needs at least 2 gold files; add documents to separate "
                 "extraction gains from boilerplate recall")

    # Confidence: the model's own probability for the tokens it chose. Rising
    # confidence is a second signal, independent of the field match rate.
    conf_base = summary["base"].get("confidence") or {}
    conf_v1 = summary[version].get("confidence") or {}
    confidence = None
    if conf_base.get("tokens") and conf_v1.get("tokens"):
        b = conf_base["mean_token_probability"]
        v = conf_v1["mean_token_probability"]
        confidence = {
            "base_mean_token_probability": b,
            "trained_mean_token_probability": v,
            "delta": round(v - b, 6),
            "improved": v > b,
            "base_low_confidence_fraction": conf_base.get("low_confidence_fraction"),
            "trained_low_confidence_fraction": conf_v1.get("low_confidence_fraction"),
        }
        reasons.append(
            f"confidence {'rose' if v > b else 'did not rise'}: mean token probability "
            f"{b:.3f} -> {v:.3f}"
        )

    sections = section_breakdown(scores["base"], scores[version])
    report = {
        "verdict": verdict,
        "version": version,
        "document": doc_id,
        "test_held_out": held_out,
        "corpus_size": dataset_meta.get("corpus_size"),
        "training_examples": dataset_meta.get("training_examples"),
        "confidence": confidence,
        "coverage": coverage,
        "field_split": split,
        "variable_fields": sum(1 for v in variability.values() if v) if variability else None,
        "static_fields": sum(1 for v in variability.values() if not v) if variability else None,
        "sections": sections,
        "base_match_rate": base_rate,
        "trained_match_rate": v1_rate,
        "delta": round(v1_rate - base_rate, 4),
        "reasons": reasons,
        "summary": summary,
        "training": {
            "first_loss": training.get("first_loss"),
            "last_loss": training.get("last_loss"),
            "loss_decreased": training.get("loss_decreased"),
        } if training else None,
        "per_field": {name: scores[name]["fields"] for name in models},
        "list_counts": {name: scores[name]["list_counts"] for name in models},
        "missing_fields": {name: scores[name]["missing_fields"] for name in models},
        "extra_fields": {name: scores[name]["extra_fields"] for name in models},
    }
    # The comparison belongs to the version it judged, so v1's verdict never
    # overwrites v2's.
    out = results_dir_for(cfg, version, doc_id) / "comparison.json"
    write_json(out, report)

    if not args.quiet:
        print_sections(sections)
        print_table(scores["base"], scores[version], args.max_rows, not args.all_fields)
    print()
    print(f"document        : {doc_id}"
          + ("  (held out of training)" if held_out else "  (included in training)"))
    print(f"trained on      : {dataset_meta.get('training_examples', '?')} document(s) "
          f"of {dataset_meta.get('corpus_size', '?')}")
    print(f"base match rate : {base_rate:.1%}  ({scores['base']['matched_fields']}/{scores['base']['total_fields']} fields)")
    trained = scores[version]
    print(f"{version:<4} match rate : {v1_rate:.1%}  "
          f"({trained['matched_fields']}/{trained['total_fields']} fields)")
    print(f"delta           : {v1_rate - base_rate:+.1%}")
    if split:
        for bucket, label in (("variable", "extraction"), ("static", "boilerplate")):
            b = split["base"].get(bucket, {})
            v = split[version].get(bucket, {})
            if b.get("match_rate") is None:
                continue
            print(f"  {bucket + ' fields':16} ({label:11}): "
                  f"{b['match_rate']:.1%} -> {v['match_rate']:.1%}  "
                  f"({b['total']} field(s))")
    if coverage.get("base") and coverage.get(version):
        b, v = coverage["base"], coverage[version]
        g = coverage.get("gold", {})
        print(f"OCR coverage    : numbers {b['number_coverage']:.1%} -> {v['number_coverage']:.1%}"
              + (f"  (gold {g['number_coverage']:.1%})" if g.get("number_coverage") else ""))
        print(f"                  words   {b['word_coverage']:.1%} -> {v['word_coverage']:.1%}"
              + (f"  (gold {g['word_coverage']:.1%})" if g.get("word_coverage") else ""))
    if confidence:
        print(f"confidence      : {confidence['base_mean_token_probability']:.3f} -> "
              f"{confidence['trained_mean_token_probability']:.3f} "
              f"({confidence['delta']:+.3f} mean token probability)")
    for reason in reasons:
        print(f"  - {reason}")
    print(f"\nVERDICT: {verdict}")
    print(f"written: {out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
