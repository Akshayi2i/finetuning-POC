"""Shared helpers: config loading, path resolution, logging, JSON IO.

Every path in config.yaml is relative to the project root (the directory that
holds config.yaml), so scripts can be run from anywhere.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


def pymupdf():
    """PyMuPDF, imported under whichever name this version provides.

    The package renamed its module from `fitz` to `pymupdf`; importing the old
    name still works but prints a deprecation warning on every run.
    """
    try:
        import pymupdf as _pymupdf
        return _pymupdf
    except ImportError:
        import fitz as _pymupdf
        return _pymupdf


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger(name)


def load_config(path: str | Path | None = None) -> dict:
    cfg_file = Path(path) if path else DEFAULT_CONFIG
    if not cfg_file.is_absolute():
        cfg_file = ROOT / cfg_file
    with open(cfg_file, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_file"] = str(cfg_file)
    return cfg


def resolve(rel: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)


def cfg_path(cfg: dict, key: str) -> Path:
    """Look up paths.<key> from the config and resolve it."""
    try:
        return resolve(cfg["paths"][key])
    except KeyError as exc:
        raise KeyError(f"paths.{key} missing from {cfg.get('_config_file')}") from exc


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text(p: Path) -> str:
    return Path(p).read_text(encoding="utf-8")


def write_text(p: Path, text: str) -> None:
    p = Path(p)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8", newline="\n")


def read_json(p: Path) -> Any:
    return json.loads(read_text(p))


def write_json(p: Path, obj: Any, indent: int = 2) -> None:
    write_text(p, json.dumps(obj, indent=indent, ensure_ascii=False) + "\n")


def compact_json(obj: Any) -> str:
    """Single-line JSON, the exact string the model is trained to emit."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def set_vision_env(cfg: dict) -> int:
    """Pin the Qwen-VL visual token budget for every stage.

    ms-swift and qwen_vl_utils both read MAX_PIXELS when deciding how to resize a
    page, so training and inference must set the same value or the image is
    tokenized differently on each side - modality drift the prompt fingerprint
    cannot see.
    """
    import os

    long_side = int(cfg["model"]["max_image_long_side_px"])
    os.environ["MAX_PIXELS"] = str(long_side * long_side)
    return long_side * long_side


# --------------------------------------------------------------------------- corpus


# Suffixes labellers commonly append to the gold file's name.
GOLD_SUFFIXES = (
    "extraction", "extracted", "gold", "golden", "groundtruth", "gt",
    "label", "labels", "labelled", "labeled", "annotation", "annotations",
    "expected", "output", "json",
)


def name_key(stem: str) -> str:
    """Normalize a file stem so a PDF and its gold JSON collapse to the same key.

    'Signed Application - Client 6'          -> 'signedapplicationclient6'
    'Signed_Application_Client_6_extraction' -> 'signedapplicationclient6'
    """
    key = re.sub(r"[^a-z0-9]+", "", stem.lower())
    changed = True
    while changed:
        changed = False
        for suffix in GOLD_SUFFIXES:
            if key.endswith(suffix) and len(key) > len(suffix):
                key = key[: -len(suffix)]
                changed = True
    return key


def slugify(stem: str) -> str:
    """Filesystem-safe id used for this document's OCR and result folders."""
    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return slug or "document"


class Document:
    """One training/test item: a PDF and the gold JSON that labels it."""

    __slots__ = ("doc_id", "pdf", "gold")

    def __init__(self, doc_id: str, pdf: Path, gold: Path):
        self.doc_id = doc_id
        self.pdf = pdf
        self.gold = gold

    def __repr__(self) -> str:
        return f"Document({self.doc_id})"


def discover_documents(data_dir: Path) -> tuple[list[Document], list[Path], list[Path]]:
    """Pair every PDF under data_dir with its gold JSON, wherever they are filed.

    Matching is by normalized file stem, so 'training sample/Doc 6.pdf' pairs with
    'golden json/Doc_6_extraction.json'. Returns (documents, unpaired_pdfs,
    unpaired_golds) - the caller decides how loudly to complain.
    """
    data_dir = Path(data_dir)
    pdfs = sorted(p for p in data_dir.rglob("*.pdf") if not p.name.startswith("."))
    golds = sorted(p for p in data_dir.rglob("*.json") if not p.name.startswith("."))

    by_key: dict[str, list[Path]] = {}
    for gold in golds:
        by_key.setdefault(name_key(gold.stem), []).append(gold)

    documents: list[Document] = []
    unpaired_pdfs: list[Path] = []
    matched: set[Path] = set()
    for pdf in pdfs:
        candidates = by_key.get(name_key(pdf.stem), [])
        if not candidates:
            unpaired_pdfs.append(pdf)
            continue
        gold = candidates[0]
        matched.add(gold)
        documents.append(Document(slugify(pdf.stem), pdf, gold))

    documents.sort(key=lambda d: d.doc_id)
    return documents, unpaired_pdfs, [g for g in golds if g not in matched]


def ocr_dir_for(cfg: dict, doc_id: str) -> Path:
    """Per-document OCR folder: outputs/ocr/<doc_id>/page_N.png + page_N.md."""
    return cfg_path(cfg, "ocr_dir") / doc_id


_PAGE_RE = re.compile(r"^page_(\d+)$")


def page_assets(ocr_dir: Path) -> list[tuple[Path, Path]]:
    """Return [(png, md), ...] ordered by page number, for pages that have both."""
    ocr_dir = Path(ocr_dir)
    pages: list[tuple[int, Path, Path]] = []
    for png in ocr_dir.glob("page_*.png"):
        m = _PAGE_RE.match(png.stem)
        if not m:
            continue
        md = png.with_suffix(".md")
        if md.exists():
            pages.append((int(m.group(1)), png, md))
    pages.sort(key=lambda t: t[0])
    return [(png, md) for _, png, md in pages]


def require_pages(ocr_dir: Path) -> list[tuple[Path, Path]]:
    pages = page_assets(ocr_dir)
    if not pages:
        raise FileNotFoundError(
            f"No page_N.png + page_N.md pairs in {ocr_dir}. Run src/run_ocr.py first."
        )
    return pages


def extract_json_object(text: str) -> tuple[Any | None, str]:
    """Pull the first complete JSON object out of raw model output.

    Returns (parsed_or_None, reason). Tolerates code fences and trailing prose,
    which the prompt forbids but models still occasionally emit.
    """
    if text is None:
        return None, "empty output"
    cleaned = text.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned), "clean parse"
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1:
        return None, "no '{' in output"

    depth, in_str, escaped = 0, False, False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == chr(92):
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    return json.loads(candidate), "recovered balanced object"
                except json.JSONDecodeError as exc:
                    return None, f"balanced object did not parse: {exc}"
    return None, "unterminated JSON object"


def repair_truncated_json(text: str) -> tuple[Any | None, str]:
    """Recover the complete prefix of a JSON object that was cut off mid-generation.

    A model that hits max_new_tokens emits a valid *prefix*, which json.loads
    rejects outright - so a 99%-complete answer would otherwise score zero
    fields. This rewinds to the last position where every container could be
    closed cleanly, closes them, and parses that.

    Returns (parsed_or_None, reason). The result is a strict subset of what the
    model produced: no value is invented, the tail is dropped.
    """
    if not text:
        return None, "empty output"
    start = text.find("{")
    if start == -1:
        return None, "no '{' in output"

    stack: list[str] = []
    safe: list[tuple[int, tuple[str, ...]]] = []  # (cut index, open containers)
    in_str = escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == chr(92):
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None, "mismatched brackets"
            stack.pop()
            safe.append((i + 1, tuple(stack)))
            if not stack:
                break
        elif ch == "," and not in_str:
            safe.append((i, tuple(stack)))  # cut before the comma

    # Newest safe points first: keep as much of the answer as parses.
    for cut, open_containers in reversed(safe[-2000:]):
        candidate = text[start:cut] + "".join(reversed(open_containers))
        try:
            return json.loads(candidate), (
                f"repaired truncated JSON: kept {cut - start} of "
                f"{len(text) - start} chars, closed {len(open_containers)} container(s)"
            )
        except json.JSONDecodeError:
            continue
    return None, "could not repair truncated JSON"


def recover_json_object(text: str) -> tuple[Any | None, str, bool]:
    """Parse model output, falling back to truncation repair.

    Returns (parsed_or_None, reason, repaired). `repaired` True means the output
    was NOT valid JSON and only a prefix was recovered - callers must keep
    reporting it as invalid while still scoring the fields it did contain.
    """
    parsed, reason = extract_json_object(text)
    if parsed is not None:
        return parsed, reason, False
    parsed, repair_reason = repair_truncated_json(text or "")
    if parsed is not None:
        return parsed, f"{reason}; {repair_reason}", True
    return None, f"{reason}; {repair_reason}", False


def validate_against_schema(obj: Any, schema_path: Path) -> tuple[bool, list[str]]:
    """Best-effort jsonschema validation. Absent jsonschema/schema => (True, [])."""
    schema_path = Path(schema_path)
    if not schema_path.exists():
        return True, []
    try:
        import jsonschema
    except ImportError:
        return True, []
    validator = jsonschema.Draft202012Validator(read_json(schema_path))
    errors = [
        f"{'.'.join(str(x) for x in e.path) or '<root>'}: {e.message}"
        for e in sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
    ]
    return (not errors), errors
