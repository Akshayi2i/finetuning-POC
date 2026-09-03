"""Stage 3/5 - run the extraction with either the base model or merged v1.

Runs on the test document by default. Both runs use the identical system prompt,
images and OCR text (built by prompting.build_messages) and greedy decoding, so
the only difference between the two outputs is the weights.

Alongside the JSON this records a confidence score: the model's own probability
for each token it chose. Rising confidence on the same document is a second,
independent signal that training took effect.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Document,
    cfg_path,
    ensure_dir,
    is_version,
    load_config,
    merged_dir_for,
    name_key,
    read_json,
    recover_json_object,
    results_dir_for,
    set_vision_env,
    setup_logging,
    validate_against_schema,
    write_json,
    write_text,
)
from common import discover_documents  # noqa: E402
from modeling import load_model_and_processor  # noqa: E402
from prompting import build_messages, conversation_fingerprint  # noqa: E402
from run_ocr import ensure_ocr, ocr_document  # noqa: E402

log = setup_logging("infer")


def resolve_model_path(cfg: dict, which: str) -> tuple[str, str | None]:
    """Return (path_or_id, revision) for 'base' or for a version like 'v1'."""
    if which == "base":
        return cfg["model"]["base_model_id"], cfg["model"].get("revision")
    if not is_version(which):
        raise SystemExit(f"--model must be 'base' or a version like v1/v2, got {which!r}")
    merged = merged_dir_for(cfg, which)
    if not any(merged.glob("*.safetensors")):
        raise FileNotFoundError(
            f"{merged} holds no merged weights. Run src/merge.py --version {which} first."
        )
    return str(merged), None


def weight_signature(model) -> float | None:
    """Cheap fingerprint of the language-model weights.

    base and v1 must produce different values; identical values mean v1 loaded
    the base weights (a named failure point in the spec).
    """
    import torch

    total = 0.0
    sampled = 0
    for name, param in model.named_parameters():
        if "visual" in name or "vision" in name:
            continue  # the ViT is frozen: it is identical in base and v1 by design
        if name.endswith(("q_proj.weight", "o_proj.weight", "down_proj.weight")) and param.dim() == 2:
            with torch.no_grad():
                total += float(param.detach().flatten()[:4096].float().sum().item())
            sampled += 1
            if sampled >= 12:
                break
    if not sampled:
        log.warning("no LoRA-targeted weights matched; weight signature unavailable")
        return None
    return round(total, 6)


def make_confidence_recorder():
    """LogitsProcessor that records the log-probability of each chosen token.

    Only correct for greedy decoding, where the chosen token is the argmax of the
    final scores. Storing one float per step keeps this negligible in memory,
    unlike output_scores=True which would materialise every step's full vocab.
    """
    import torch
    from transformers import LogitsProcessor

    class ConfidenceRecorder(LogitsProcessor):
        def __init__(self) -> None:
            self.logprobs: list[float] = []

        def __call__(self, input_ids, scores):
            with torch.no_grad():
                logprobs = torch.log_softmax(scores[0].float(), dim=-1)
                self.logprobs.append(float(logprobs.max()))
            return scores

    return ConfidenceRecorder()


def confidence_stats(logprobs: list[float]) -> dict:
    """Summarise per-token confidence into numbers that can be compared run to run."""
    import math

    if not logprobs:
        return {"tokens": 0}
    probs = sorted(math.exp(lp) for lp in logprobs)
    mean_logprob = sum(logprobs) / len(logprobs)

    def percentile(p: float) -> float:
        idx = min(len(probs) - 1, max(0, int(round(p * (len(probs) - 1)))))
        return round(probs[idx], 6)

    return {
        "tokens": len(logprobs),
        # exp(mean log p): the geometric-mean per-token probability
        "mean_token_probability": round(math.exp(mean_logprob), 6),
        "mean_token_logprob": round(mean_logprob, 6),
        "min_token_probability": round(probs[0], 6),
        "p10_token_probability": percentile(0.10),
        "median_token_probability": percentile(0.50),
        "low_confidence_tokens": sum(1 for p in probs if p < 0.5),
        "low_confidence_fraction": round(sum(1 for p in probs if p < 0.5) / len(probs), 4),
    }


def prepare_inputs(processor, messages: list[dict], model):
    """Apply the chat template and attach the page images."""
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
    except Exception as exc:  # qwen_vl_utils missing or unusable
        log.warning("qwen_vl_utils unavailable (%s); loading images with PIL", exc)
        from PIL import Image

        image_inputs = [
            Image.open(block["image"]).convert("RGB")
            for msg in messages
            if isinstance(msg["content"], list)
            for block in msg["content"]
            if block.get("type") == "image"
        ]
        video_inputs = None

    if not image_inputs:
        raise RuntimeError("no images were prepared: the ocr_plus_image modality would be broken")
    log.info("prepared %d image(s) for the prompt", len(image_inputs))

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    return inputs.to(model.device)


def resolve_doc(cfg: dict, requested: str | None) -> tuple[str, str | None]:
    """Which document to extract from, and the fingerprint recorded for it."""
    meta_file = cfg_path(cfg, "dataset").parent / "dataset_meta.json"
    meta = read_json(meta_file) if meta_file.exists() else {}
    if requested:
        return requested, next(
            (d.get("fingerprint") for d in meta.get("documents", [])
             if d.get("doc_id") == requested), None)
    doc_id = meta.get("test_document")
    if not doc_id:
        raise SystemExit(
            "no test document recorded. Run src/build_dataset.py first, or pass --doc <doc_id>."
        )
    return doc_id, meta.get("test_fingerprint")


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract JSON from a document with base or v1.")
    ap.add_argument("--model", required=True,
                    help="'base', or a fine-tuned version like v1 / v2")
    ap.add_argument("--config", default=None)
    ap.add_argument("--doc", default=None, help="doc_id to extract (default: the test document)")
    ap.add_argument("--pdf", default=None,
                    help="path to any PDF to extract, including one outside the corpus. "
                         "Use instead of --doc; no gold JSON is needed.")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="load in 4-bit for inference on a small GPU (slightly changes outputs)")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--no-ocr", action="store_true",
                    help="do not OCR the document first; use existing OCR output")
    args = ap.parse_args()

    cfg = load_config(args.config)
    inference = cfg["inference"]
    if inference.get("mode") != "ocr_plus_image":
        log.error("inference.mode must be ocr_plus_image to match training, got %r",
                  inference.get("mode"))
        return 2

    if args.pdf and args.doc:
        log.error("pass either --pdf or --doc, not both")
        return 2

    # One command does the whole extraction: OCR the document with MinerU if it has
    # not been done, then run it through the model you asked for.
    if args.pdf:
        # An arbitrary PDF, which is the production case: no gold, no corpus entry.
        pdf_path = Path(args.pdf).expanduser()
        if not pdf_path.is_file():
            log.error("no such PDF: %s", pdf_path)
            return 2
        doc_id, expected_fp = name_key(pdf_path.stem) or "adhoc", None
        log.info("ad-hoc extraction of %s as doc_id=%s", pdf_path.name, doc_id)
        if not args.no_ocr:
            engine = cfg.get("ocr", {}).get("engine", "mineru")
            ocr_document(Document(doc_id, pdf_path.resolve(), Path("(none)")), cfg, engine)
    else:
        doc_id, expected_fp = resolve_doc(cfg, args.doc)
        if not args.no_ocr:
            documents, _, _ = discover_documents(cfg_path(cfg, "data_dir"))
            target = [d for d in documents if d.doc_id == doc_id]
            if target:
                ensure_ocr(cfg, target)
            else:
                log.warning("%s is not in %s; using whatever OCR output exists",
                            doc_id, cfg_path(cfg, "data_dir"))

    max_pixels = set_vision_env(cfg)
    log.info("document=%s model=%s MAX_PIXELS=%d", doc_id, args.model, max_pixels)

    model_path, revision = resolve_model_path(cfg, args.model)
    log.info("model=%s resolved to %s", args.model, model_path)

    messages, images = build_messages(cfg, doc_id, assistant_text=None)
    fingerprint = conversation_fingerprint(messages)
    if expected_fp and expected_fp != fingerprint:
        log.error("PROMPT/MODALITY DRIFT: dataset built %s with %s, inference uses %s",
                  doc_id, expected_fp, fingerprint)
        return 3

    import torch

    model, processor = load_model_and_processor(
        model_path, cfg, load_in_4bit=args.load_in_4bit,
        device_map=args.device_map, revision=revision,
    )
    model.eval()
    signature = weight_signature(model)
    log.info("weight signature: %s", signature)

    inputs = prepare_inputs(processor, messages, model)
    prompt_len = int(inputs["input_ids"].shape[-1])
    log.info("prompt length: %d tokens", prompt_len)

    temperature = float(inference.get("temperature", 0.0))
    gen_kwargs = {"max_new_tokens": int(inference.get("max_new_tokens", 2048))}
    recorder = None
    if temperature <= 0:
        gen_kwargs["do_sample"] = False
        recorder = make_confidence_recorder()
        gen_kwargs["logits_processor"] = [recorder]
    else:
        gen_kwargs.update({"do_sample": True, "temperature": temperature})
        log.warning("temperature > 0: confidence is only recorded for greedy decoding")

    log.info("generating (max_new_tokens=%d)", gen_kwargs["max_new_tokens"])
    with torch.inference_mode():
        generated = model.generate(**inputs, **gen_kwargs)
    new_tokens = generated[0][prompt_len:]
    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
    log.info("generated %d tokens, %d chars", len(new_tokens), len(raw))

    confidence = confidence_stats(recorder.logprobs if recorder else [])
    if confidence.get("tokens"):
        log.info("confidence: mean token probability %.4f, %d low-confidence token(s) (<0.5)",
                 confidence["mean_token_probability"], confidence["low_confidence_tokens"])

    # Per-model, per-version folder so base / v1 / v2 never overwrite each other,
    # and compare.py can find each one:
    #   results/base model results/<doc_id>/
    #   results/trained model results/v1/<doc_id>/
    results_dir = ensure_dir(results_dir_for(cfg, args.model, doc_id))
    write_text(results_dir / "raw.txt", raw + "\n")

    parsed, reason, repaired = recover_json_object(raw)
    hit_cap = int(len(new_tokens)) >= int(inference.get("max_new_tokens", 2048))
    schema_ok, schema_errors = (False, ["output did not parse"])
    if parsed is not None:
        write_json(results_dir / "output.json", parsed)
        schema_ok, schema_errors = validate_against_schema(parsed, cfg_path(cfg, "json_schema"))
        if repaired:
            # Not valid JSON: only a prefix was recovered. compare.py keeps reporting
            # json_valid_strict=false while still scoring the fields it did contain.
            log.error("output was NOT valid JSON (%s)", reason)
            if hit_cap:
                log.error("generation stopped at max_new_tokens=%d: raise it above the "
                          "gold answer length", gen_kwargs["max_new_tokens"])
        else:
            log.info("JSON parsed (%s); schema_ok=%s", reason, schema_ok)
        for err in schema_errors[:5]:
            log.warning("  schema: %s", err)
    else:
        stale = results_dir / "output.json"
        if stale.exists():
            stale.unlink()
        log.error("output is not valid JSON (%s); raw text kept in %s", reason, results_dir / "raw.txt")

    write_json(results_dir / "meta.json", {
        "model": args.model,
        "doc_id": doc_id,
        "model_path": model_path,
        "revision": revision,
        "load_in_4bit": args.load_in_4bit,
        "weight_signature": signature,
        "conversation_fingerprint": fingerprint,
        "images": images,
        "prompt_tokens": prompt_len,
        "generated_tokens": int(len(new_tokens)),
        "parsed": parsed is not None,
        "json_valid_strict": parsed is not None and not repaired,
        "repaired_from_truncation": repaired,
        "hit_max_new_tokens": hit_cap,
        "parse_note": reason,
        "schema_ok": schema_ok,
        "schema_errors": schema_errors[:20],
        "confidence": confidence,
        "generation": {k: v for k, v in gen_kwargs.items() if k != "logits_processor"},
    })

    return 0 if (parsed is not None and not repaired) else 4


if __name__ == "__main__":
    raise SystemExit(main())
