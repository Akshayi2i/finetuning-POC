"""Stage 2 - assemble the training corpus, one chat-format example per document.

Writes two encodings of the same conversations:

  training dataset/dataset/train.jsonl        the spec's chat format (content blocks)
  training dataset/dataset/train_swift.jsonl  ms-swift native form (<image> tags + images list)

The test document is held out of training by default (corpus.include_test_in_training),
so a gain on it is evidence the model learned the document *type* rather than
memorised the answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Document,
    cfg_path,
    compact_json,
    discover_documents,
    ensure_dir,
    load_config,
    ocr_dir_for,
    page_assets,
    read_json,
    read_text,
    setup_logging,
    validate_against_schema,
    write_json,
    write_text,
)
from prompting import (  # noqa: E402
    build_messages,
    conversation_fingerprint,
    to_swift_record,
)
from run_ocr import ensure_ocr  # noqa: E402

log = setup_logging("build_dataset")


def image_tokens(images: list[str]) -> int:
    """Visual tokens Qwen3-VL charges for these page images.

    patch_size 16 with a 2x2 spatial merge, so a page costs (W/16)*(H/16)/4 tokens.
    """
    from PIL import Image

    total = 0
    for path in images:
        with Image.open(path) as img:
            width, height = img.size
        total += (width // 16) * (height // 16) // 4
    return total


def pick_test_document(cfg: dict, documents: list[Document]) -> Document:
    """Resolve corpus.test_document, defaulting to the first document."""
    from common import name_key

    wanted = (cfg.get("corpus") or {}).get("test_document")
    if not wanted:
        return documents[0]
    key = name_key(str(wanted))
    for doc in documents:
        if doc.doc_id == wanted or name_key(doc.doc_id) == key:
            return doc
    raise SystemExit(
        f"corpus.test_document={wanted!r} matches no document. Available: "
        + ", ".join(d.doc_id for d in documents)
    )


class TokenCounter:
    """Counts tokens per example, loading the processor at most once."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.processor = None
        self.exact = True
        self.available = True

    def _load(self) -> None:
        from common import set_vision_env

        set_vision_env(self.cfg)
        try:
            from transformers import AutoProcessor
        except ImportError:
            log.warning("transformers not installed; token counts unavailable")
            self.available = False
            return
        model_id = self.cfg["model"]["base_model_id"]
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, revision=self.cfg["model"].get("revision"), trust_remote_code=True)
        except Exception as exc:
            log.warning("could not load processor for %s (%s); token counts unavailable",
                        model_id, exc)
            self.available = False

    def count(self, messages: list[dict], images: list[str]) -> int | None:
        if self.processor is None and self.available:
            self._load()
        if not self.available:
            return None
        try:
            from PIL import Image

            pil = [Image.open(p).convert("RGB") for p in images]
            text = self.processor.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=False)
            batch = self.processor(text=[text], images=pil or None, return_tensors="pt")
            return int(batch["input_ids"].shape[-1])
        except Exception as exc:
            if self.exact:
                log.warning("exact token count unavailable (%s); estimating instead", exc)
                self.exact = False
            try:
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
                text = self.processor.apply_chat_template(messages, tokenize=False,
                                                          add_generation_prompt=False)
                return len(tokenizer(text)["input_ids"]) + image_tokens(images)
            except Exception as exc2:
                log.error("TOKEN COUNT FAILED (%s): max_seq_length is UNVERIFIED", exc2)
                self.available = False
                return None


def check_ocr(cfg: dict, doc: Document) -> tuple[bool, list[int]]:
    """OCR text is the primary source; report pages that produced none."""
    pages = page_assets(ocr_dir_for(cfg, doc.doc_id))
    chars = [len(read_text(md).strip()) for _, md in pages]
    return bool(pages) and any(n >= 20 for n in chars), chars


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the training corpus JSONL.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-token-count", action="store_true", help="skip loading the processor")
    ap.add_argument("--no-ocr", action="store_true",
                    help="do not OCR missing documents first (fail instead)")
    ap.add_argument("--force-ocr", action="store_true", help="re-OCR every document")
    args = ap.parse_args()

    cfg = load_config(args.config)
    corpus_cfg = cfg.get("corpus") or {}
    documents, unpaired_pdfs, _ = discover_documents(cfg_path(cfg, "data_dir"))
    if not documents:
        log.error("no PDF+gold pairs under %s", cfg_path(cfg, "data_dir"))
        return 2
    for pdf in unpaired_pdfs:
        log.error("PDF with no gold JSON, excluded: %s", pdf)

    # Stage 1 runs itself: any document without OCR output is OCR'd now, so the
    # corpus can be built with one command instead of two.
    if not args.no_ocr:
        failed = ensure_ocr(cfg, documents, force=args.force_ocr)
        if failed:
            log.error("OCR failed for %s; those documents cannot be used", failed)

    test_doc = pick_test_document(cfg, documents)
    include_test = bool(corpus_cfg.get("include_test_in_training", False))
    log.info("corpus: %d document(s); test document = %s (%s)",
             len(documents), test_doc.doc_id,
             "also trained on" if include_test else "HELD OUT of training")

    records, swift_records, per_doc = [], [], []
    skipped = []
    max_len = int(cfg["training"]["max_seq_length"])
    counter = None if args.no_token_count else TokenCounter(cfg)

    for doc in documents:
        is_test = doc.doc_id == test_doc.doc_id
        if is_test and not include_test:
            log.info("[%s] held out as the test document", doc.doc_id)

        try:
            gold = read_json(doc.gold)
        except Exception as exc:
            log.error("[%s] gold JSON does not parse (%s); excluded", doc.doc_id, exc)
            skipped.append({"doc_id": doc.doc_id, "reason": f"gold does not parse: {exc}"})
            continue

        ok, errors = validate_against_schema(gold, cfg_path(cfg, "json_schema"))
        if not ok:
            log.warning("[%s] gold does not match prompts/schema.json (%d issue(s), first: %s)",
                        doc.doc_id, len(errors), errors[0] if errors else "")

        has_text, chars = check_ocr(cfg, doc)
        if not chars:
            log.error("[%s] no OCR output; run src/run_ocr.py first. Excluded.", doc.doc_id)
            skipped.append({"doc_id": doc.doc_id, "reason": "no OCR output"})
            continue
        if not has_text:
            log.error("[%s] OCR produced no text on any page; the example would be "
                      "image-only. Excluded. For scans use --engine mineru.", doc.doc_id)
            skipped.append({"doc_id": doc.doc_id, "reason": "OCR text empty"})
            continue

        assistant_text = compact_json(gold)
        messages, images = build_messages(cfg, doc.doc_id, assistant_text=assistant_text)
        assert messages[-1]["content"] == assistant_text, "assistant turn must be the gold JSON"
        swift_record = to_swift_record(messages, images)
        n_tags = swift_record["messages"][1]["content"].count("<image>")
        assert n_tags == len(images), f"{n_tags} <image> tags for {len(images)} images"

        tokens = counter.count(messages, images) if counter else None
        fits = None if tokens is None else tokens <= max_len
        if tokens is not None and not fits:
            log.error("[%s] example is %d tokens, over max_seq_length=%d. ms-swift is set to "
                      "DROP over-long examples, so this document would train on nothing. "
                      "Raise max_seq_length or lower max_image_long_side_px.",
                      doc.doc_id, tokens, max_len)

        info = {
            "doc_id": doc.doc_id,
            "pdf": str(doc.pdf),
            "gold": str(doc.gold),
            "pages": len(images),
            "ocr_chars": sum(chars),
            "empty_ocr_pages": [i for i, n in enumerate(chars, start=1) if n < 20],
            "gold_chars": len(assistant_text),
            "tokens": tokens,
            "fits_max_seq_length": fits,
            "role": "test" if is_test else "train",
            "in_training": include_test or not is_test,
            "fingerprint": conversation_fingerprint(messages),
        }
        per_doc.append(info)
        log.info("[%s] %d page(s), %d OCR chars, gold %d chars%s -> %s",
                 doc.doc_id, len(images), sum(chars), len(assistant_text),
                 f", {tokens} tokens" if tokens else "", info["role"])

        if info["in_training"]:
            records.append({"messages": messages})
            swift_records.append(swift_record)

    # The test document is the measurement. If it was excluded, every number the POC
    # reports would come from a degraded input, so fail rather than quietly continue.
    if any(sk["doc_id"] == test_doc.doc_id for sk in skipped):
        reason = next(sk["reason"] for sk in skipped if sk["doc_id"] == test_doc.doc_id)
        log.error("THE TEST DOCUMENT %s IS UNUSABLE (%s). base and v1 would both be measured "
                  "on a broken input. Fix it, or point corpus.test_document at another "
                  "document.", test_doc.doc_id, reason)
        return 3

    if not records:
        if len(documents) == 1 and not include_test:
            log.error(
                "the corpus holds only %s, and it is held out as the test document, so there "
                "is nothing to train on. Either add more documents to %s (synthetic variants "
                "of the same type, each with its own gold JSON), or set "
                "corpus.include_test_in_training: true to train on this one document - which "
                "measures memorisation rather than learning.",
                test_doc.doc_id, cfg_path(cfg, "data_dir"))
        else:
            log.error("no trainable examples were produced from %d document(s); see the "
                      "exclusions above", len(documents))
        return 3
    minimum = int(corpus_cfg.get("min_train_documents", 1))
    if len(records) < minimum:
        log.error("only %d training document(s), corpus.min_train_documents=%d",
                  len(records), minimum)
        return 3

    dataset = cfg_path(cfg, "dataset")
    ensure_dir(dataset.parent)
    write_text(dataset, "\n".join(compact_json(r) for r in records) + "\n")
    write_text(cfg_path(cfg, "swift_dataset"),
               "\n".join(compact_json(r) for r in swift_records) + "\n")
    log.info("wrote %d training example(s) to %s", len(records), dataset)

    trained = [d for d in per_doc if d["in_training"]]
    token_values = [d["tokens"] for d in trained if d["tokens"]]
    if token_values:
        log.info("training example lengths: min %d / max %d / total %d tokens (limit %d each)",
                 min(token_values), max(token_values), sum(token_values), max_len)
        over = [d["doc_id"] for d in trained if d["fits_max_seq_length"] is False]
        if over:
            log.error("these documents exceed max_seq_length and would be dropped: %s", over)

    test_info = next((d for d in per_doc if d["doc_id"] == test_doc.doc_id), None)
    write_json(dataset.parent / "dataset_meta.json", {
        "corpus_size": len(documents),
        "training_examples": len(records),
        "test_document": test_doc.doc_id,
        "test_held_out": not include_test,
        "test_fingerprint": test_info["fingerprint"] if test_info else None,
        "max_seq_length": max_len,
        "documents": per_doc,
        "skipped": skipped,
    })
    log.info("test document %s: %s", test_doc.doc_id,
             "held out" if not include_test else "included in training")
    if skipped:
        log.warning("%d document(s) excluded: %s", len(skipped),
                    ", ".join(s["doc_id"] for s in skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
