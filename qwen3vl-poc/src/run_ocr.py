"""Stage 1 - OCR every document in the corpus.

For each PDF found under paths.data_dir:
    outputs/ocr/<doc_id>/page_{n}.png   page image, capped at max_image_long_side_px
    outputs/ocr/<doc_id>/page_{n}.md    the text MinerU read off that page

Page images are always rendered with PyMuPDF so the resolution cap is applied
exactly. The text comes from a swappable engine:

  mineru  (default) - MinerU 2.x CLI, layout-aware markdown per page
  pymupdf           - embedded text layer only; no OCR. Dry-run fallback, and
                      useless on a scanned PDF, which has no text layer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Document,
    cfg_path,
    discover_documents,
    ensure_dir,
    load_config,
    ocr_dir_for,
    pymupdf,
    setup_logging,
    write_json,
    write_text,
)

log = setup_logging("run_ocr")


# --------------------------------------------------------------------------- images


def render_pages(pdf: Path, out_dir: Path, dpi: int, max_long_side: int) -> list[Path]:
    """Render every PDF page to PNG, downscaled so long side <= max_long_side."""
    fitz = pymupdf()

    written = []
    with fitz.open(pdf) as doc:
        for page_index, page in enumerate(doc, start=1):
            zoom = dpi / 72.0
            rect = page.rect
            long_side_pt = max(rect.width, rect.height)
            if long_side_pt * zoom > max_long_side:
                zoom = max_long_side / long_side_pt
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            png = out_dir / f"page_{page_index}.png"
            pix.save(png)
            if max(pix.width, pix.height) > max_long_side:
                raise RuntimeError(
                    f"page {page_index} exceeds max_image_long_side_px "
                    f"({max(pix.width, pix.height)} > {max_long_side})"
                )
            written.append(png)
    return written


def page_count(pdf: Path) -> int:
    fitz = pymupdf()

    with fitz.open(pdf) as doc:
        return doc.page_count


# --------------------------------------------------------------------------- MinerU


def _md_from_content_list(items: list[dict], n_pages: int) -> list[str]:
    """Group MinerU content_list blocks by page_idx and render markdown per page."""
    per_page: list[list[str]] = [[] for _ in range(n_pages)]
    for block in items:
        idx = int(block.get("page_idx", 0))
        if not 0 <= idx < n_pages:
            continue
        kind = block.get("type")
        if kind == "text":
            text = (block.get("text") or "").strip()
            if not text:
                continue
            level = block.get("text_level")
            per_page[idx].append(f"{'#' * int(level)} {text}" if level else text)
        elif kind == "table":
            body = (block.get("table_body") or "").strip()
            caption = " ".join(block.get("table_caption") or []).strip()
            if caption:
                per_page[idx].append(f"**{caption}**")
            if body:
                per_page[idx].append(body)
            for note in block.get("table_footnote") or []:
                note = (note or "").strip()
                if note:
                    per_page[idx].append(note)
        elif kind == "equation":
            latex = (block.get("text") or "").strip()
            if latex:
                per_page[idx].append(latex)
        elif kind == "discarded":
            # MinerU files headers, footers and page numbers as "discarded". It also
            # misfiles real content there - on the declarations page a premium total
            # landed in it. The spec says nothing present may be skipped, so keep it.
            text = (block.get("text") or "").strip()
            if text:
                per_page[idx].append(text)
        elif kind == "image":
            caption = " ".join(block.get("image_caption") or []).strip()
            if caption:
                per_page[idx].append(f"![{caption}]()")
    return ["\n\n".join(chunks).strip() for chunks in per_page]


def run_mineru(pdf: Path, n_pages: int, backend: str, lang: str) -> list[str]:
    """Run the MinerU CLI on one PDF and return one markdown string per page."""
    exe = shutil.which("mineru") or shutil.which("magic-pdf")
    if not exe:
        raise RuntimeError(
            "MinerU CLI not found on PATH. Install it (pip install 'mineru[pipeline]') "
            "or run with --engine pymupdf for a no-OCR dry run."
        )

    with tempfile.TemporaryDirectory(prefix="mineru_") as tmp:
        tmp_dir = Path(tmp)
        cmd = [exe, "-p", str(pdf), "-o", str(tmp_dir), "-b", backend, "-l", lang]
        log.info("running: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log.error("MinerU stdout:\n%s", proc.stdout[-4000:])
            log.error("MinerU stderr:\n%s", proc.stderr[-4000:])
            raise RuntimeError(f"MinerU exited with code {proc.returncode}")

        produced = [q for q in tmp_dir.rglob("*") if q.is_file()]
        if not produced:
            # Exit code 0 with no output at all: almost always a missing model
            # download or a path MinerU silently refused. Its own log is the only
            # evidence, so surface it instead of discarding it.
            log.error("MinerU exited 0 but wrote nothing to %s", tmp_dir)
            log.error("MinerU stdout:\n%s", proc.stdout[-4000:])
            log.error("MinerU stderr:\n%s", proc.stderr[-4000:])
            raise RuntimeError("MinerU produced no output files at all (exit 0)")

        content_lists = sorted(tmp_dir.rglob("*_content_list.json"))
        if content_lists:
            items = json.loads(content_lists[0].read_text(encoding="utf-8"))
            log.info("parsed %s (%d blocks)", content_lists[0].name, len(items))
            return _md_from_content_list(items, n_pages)

        markdowns = sorted(tmp_dir.rglob("*.md"))
        if not markdowns:
            raise RuntimeError(f"MinerU produced no markdown under {tmp_dir}")
        full = markdowns[0].read_text(encoding="utf-8")
        if n_pages == 1:
            log.warning("no content_list.json; using whole-document markdown for page 1")
            return [full]
        raise RuntimeError(
            "MinerU produced no *_content_list.json, so per-page text cannot be "
            f"recovered for a {n_pages}-page PDF. Check the MinerU version."
        )


def run_pymupdf(pdf: Path) -> list[str]:
    fitz = pymupdf()

    with fitz.open(pdf) as doc:
        return [page.get_text("text").strip() for page in doc]


# --------------------------------------------------------------------------- per document


def ocr_document(doc: Document, cfg: dict, engine: str) -> dict:
    """OCR one document into outputs/ocr/<doc_id>/. Returns a summary dict."""
    out_dir = ensure_dir(ocr_dir_for(cfg, doc.doc_id))
    ocr_cfg = cfg.get("ocr", {})
    max_long_side = int(cfg["model"]["max_image_long_side_px"])

    n_pages = page_count(doc.pdf)
    log.info("[%s] %s (%d page%s)", doc.doc_id, doc.pdf.name, n_pages, "" if n_pages == 1 else "s")

    for stale in list(out_dir.glob("page_*.png")) + list(out_dir.glob("page_*.md")):
        stale.unlink()

    images = render_pages(doc.pdf, out_dir, int(ocr_cfg.get("dpi", 200)), max_long_side)

    if engine == "mineru":
        pages_md = run_mineru(
            doc.pdf, n_pages,
            ocr_cfg.get("mineru_backend", "pipeline"),
            ocr_cfg.get("mineru_lang", "en"),
        )
    else:
        pages_md = run_pymupdf(doc.pdf)

    if len(pages_md) != n_pages:
        raise RuntimeError(f"engine returned {len(pages_md)} page texts for {n_pages} pages")

    chars = []
    for i, text in enumerate(pages_md, start=1):
        write_text(out_dir / f"page_{i}.md", text.strip() + "\n")
        chars.append(len(text.strip()))

    empty = [i for i, n in enumerate(chars, start=1) if n < 20]
    if empty:
        log.warning("[%s] no text on page(s) %s", doc.doc_id, empty)
    log.info("[%s] %d page(s), %d OCR chars total", doc.doc_id, n_pages, sum(chars))

    summary = {
        "doc_id": doc.doc_id,
        "source_pdf": str(doc.pdf),
        "gold_json": str(doc.gold),
        "engine": engine,
        "pages": n_pages,
        "max_image_long_side_px": max_long_side,
        "images": [p.name for p in images],
        "chars_per_page": chars,
        "empty_pages": empty,
    }
    write_json(out_dir / "ocr_meta.json", summary)
    return summary


# --------------------------------------------------------------------------- main


def has_ocr(cfg: dict, doc_id: str) -> bool:
    """Does this document already have usable OCR output on disk?"""
    out = ocr_dir_for(cfg, doc_id)
    return (out / "ocr_meta.json").exists() and any(out.glob("page_*.md"))


def ensure_ocr(cfg: dict, documents: list, engine: str | None = None,
               force: bool = False) -> list[str]:
    """OCR any document that does not have output yet. Returns the ids that failed.

    Called by build_dataset.py and infer.py so neither needs OCR to have been run
    by hand first. Documents already done are skipped, so this is cheap to call
    on every run.
    """
    engine = engine or cfg.get("ocr", {}).get("engine", "mineru")
    todo = [d for d in documents if force or not has_ocr(cfg, d.doc_id)]
    if not todo:
        log.info("OCR already present for all %d document(s)", len(documents))
        return []
    log.info("OCR needed for %d of %d document(s), engine=%s",
             len(todo), len(documents), engine)
    failed = []
    for doc in todo:
        try:
            ocr_document(doc, cfg, engine)
        except Exception as exc:
            log.error("[%s] OCR FAILED: %s", doc.doc_id, exc)
            failed.append(doc.doc_id)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description="OCR every document in the corpus.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--engine", choices=["mineru", "pymupdf"], default=None,
                    help="overrides ocr.engine in config.yaml")
    ap.add_argument("--doc", default=None, help="OCR only this doc_id")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave documents that already have OCR output alone")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg_path(cfg, "data_dir")
    engine = args.engine or cfg.get("ocr", {}).get("engine", "mineru")

    documents, unpaired_pdfs, unpaired_golds = discover_documents(data_dir)
    for pdf in unpaired_pdfs:
        log.error("PDF with no matching gold JSON, skipped: %s", pdf)
    for gold in unpaired_golds:
        log.warning("gold JSON with no matching PDF: %s", gold)
    if not documents:
        log.error("no PDF+gold pairs found under %s. Name each gold after its PDF, e.g. "
                  "'Client 6.pdf' + 'Client_6_extraction.json'.", data_dir)
        return 2

    if args.doc:
        documents = [d for d in documents if d.doc_id == args.doc]
        if not documents:
            log.error("no document with doc_id %r", args.doc)
            return 2

    log.info("corpus: %d document(s), engine=%s", len(documents), engine)
    if engine == "pymupdf":
        log.warning("engine=pymupdf reads the embedded text layer only - a scanned PDF "
                    "will produce NO text. Use --engine mineru for scans.")

    summaries = []
    failed = []
    for doc in documents:
        if args.skip_existing and (ocr_dir_for(cfg, doc.doc_id) / "ocr_meta.json").exists():
            log.info("[%s] already has OCR output, skipping", doc.doc_id)
            continue
        try:
            summaries.append(ocr_document(doc, cfg, engine))
        except Exception as exc:
            log.error("[%s] OCR FAILED: %s", doc.doc_id, exc)
            failed.append(doc.doc_id)

    write_json(cfg_path(cfg, "ocr_dir") / "corpus_meta.json", {
        "data_dir": str(data_dir),
        "engine": engine,
        "documents": summaries,
        "failed": failed,
    })

    log.info("OCR complete: %d document(s) processed, %d failed", len(summaries), len(failed))
    if failed:
        log.error("failed documents: %s", failed)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
