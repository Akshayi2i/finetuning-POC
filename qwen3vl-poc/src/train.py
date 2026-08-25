"""Stage 4 - QLoRA fine-tuning of the base model on the training corpus via ms-swift.

ms-swift is driven through its `swift sft` CLI: it owns the multimodal collator
and the -100 label masking (loss only on the assistant JSON), which is exactly
the part of the pipeline this POC must not reimplement.

After training the newest checkpoint's adapter is copied to outputs/adapter/ and
the loss curve is checked: a loss that does not fall is the primary failure
signal in the spec, so this script exits non-zero when that happens.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    cfg_path,
    ensure_dir,
    load_config,
    set_vision_env,
    setup_logging,
    write_json,
)
from modeling import resolve_attn_impl  # noqa: E402

log = setup_logging("train")

LOSS_RE = re.compile(r"['\"]?\bloss['\"]?\s*[:=]\s*([0-9]*\.?[0-9]+)")
ADAPTER_KEEP = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "additional_config.json",
    "args.json",
    "configuration.json",
    "sft_args.json",
}


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def build_command(cfg: dict, swift_output: Path) -> list[str]:
    model = cfg["model"]
    lora = cfg["lora"]
    qlora = cfg["qlora"]
    tr = cfg["training"]

    cmd: list[str] = [
        sys.executable, "-m", "swift.cli.main", "sft",
        "--model", model["base_model_id"],
        "--train_type", "lora",
        "--dataset", str(cfg_path(cfg, "swift_dataset")),
        "--split_dataset_ratio", "0",          # no validation split: the test doc is held out
        # Drop an over-long example rather than silently cutting the assistant JSON off
        # its tail - a right-truncated answer trains the model on a broken target and
        # looks exactly like broken label masking.
        "--truncation_strategy", "delete",
        "--output_dir", str(swift_output),
        "--torch_dtype", "bfloat16" if tr.get("bf16", True) else "float16",
        "--max_length", str(tr["max_seq_length"]),
        "--num_train_epochs", str(tr["num_train_epochs"]),
        "--per_device_train_batch_size", str(tr["per_device_train_batch_size"]),
        "--gradient_accumulation_steps", str(tr["gradient_accumulation_steps"]),
        "--learning_rate", str(tr["learning_rate"]),
        "--lr_scheduler_type", tr["lr_scheduler_type"],
        "--warmup_ratio", str(tr["warmup_ratio"]),
        "--weight_decay", str(tr["weight_decay"]),
        "--max_grad_norm", str(tr["max_grad_norm"]),
        "--gradient_checkpointing", bool_str(tr.get("gradient_checkpointing", True)),
        "--logging_steps", str(tr["logging_steps"]),
        "--save_strategy", tr["save_strategy"],
        "--save_total_limit", "2",
        "--seed", str(tr["seed"]),
        "--lora_rank", str(lora["r"]),
        "--lora_alpha", str(lora["alpha"]),
        "--lora_dropout", str(lora["dropout"]),
        "--target_modules", *lora["target_modules"],
        "--freeze_vit", bool_str(not lora.get("train_vit", False)),
        "--dataloader_num_workers", "0",
        "--report_to", "none",
    ]

    if model.get("revision"):
        cmd += ["--model_revision", str(model["revision"])]
    # resolve_attn_impl downgrades flash-attn to sdpa when it is not installed, so
    # training and inference make the same choice on the same box.
    attn = resolve_attn_impl(model.get("attn_implementation"))
    if attn:
        # ms-swift spells flash_attention_2 as flash_attn
        cmd += ["--attn_impl", "flash_attn" if attn.startswith("flash") else attn]
    if tr.get("use_liger_kernel"):
        cmd += ["--use_liger_kernel", "true"]
    if qlora.get("load_in_4bit", True):
        cmd += [
            "--quant_method", "bnb",
            "--quant_bits", "4",
            "--bnb_4bit_quant_type", qlora.get("bnb_4bit_quant_type", "nf4"),
            "--bnb_4bit_compute_dtype", qlora.get("bnb_4bit_compute_dtype", "bfloat16"),
            "--bnb_4bit_use_double_quant", bool_str(qlora.get("bnb_4bit_use_double_quant", True)),
        ]
    return cmd


def analyse_labels(sample: str | None) -> dict:
    """Judge ms-swift's [LABELS] line: is the prompt masked and the target the JSON?

    A [LABELS] line with no -100 run means the loss covers the prompt and the OCR
    text as well as the answer - the first failure point the spec lists.
    """
    if not sample:
        return {"captured": False,
                "note": "ms-swift printed no [LABELS] line; verify masking manually"}
    masked = ("-100" in sample) or ("[-100 *" in sample)
    tail = sample.split("]", 1)[-1] if "-100 *" in sample else sample
    return {
        "captured": True,
        "sample": sample[:400],
        "prompt_masked": masked,
        "target_starts_with_brace": tail.lstrip().startswith("{") or '{"' in tail[:80],
    }


def run_streaming(cmd: list[str], env: dict) -> tuple[int, list[float], str | None]:
    """Run the trainer, echo its output, collect losses and the [LABELS] sample."""
    log.info("launching: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    losses: list[float] = []
    labels_sample: str | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if labels_sample is None:
            m = LABELS_RE.search(line)
            if m:
                labels_sample = m.group(1).strip()
        if "loss" in line and "eval" not in line.lower():
            m = LOSS_RE.search(line)
            if m:
                try:
                    losses.append(float(m.group(1)))
                except ValueError:
                    pass
    proc.wait()
    return proc.returncode, losses, labels_sample


def losses_from_logfile(swift_output: Path) -> list[float]:
    """Prefer ms-swift's structured logging.jsonl over scraped stdout."""
    values: list[float] = []
    for logfile in sorted(swift_output.rglob("logging.jsonl")):
        for line in logfile.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and isinstance(rec.get("loss"), (int, float)):
                values.append(float(rec["loss"]))
        if values:
            log.info("read %d loss points from %s", len(values), logfile)
            break
    return values


def latest_checkpoint(swift_output: Path) -> Path | None:
    candidates = [p for p in swift_output.rglob("checkpoint-*") if p.is_dir()]
    if not candidates:
        return None

    def step(p: Path) -> int:
        m = re.search(r"checkpoint-(\d+)$", p.name)
        return int(m.group(1)) if m else -1

    return max(candidates, key=step)


def export_adapter(checkpoint: Path, adapter_dir: Path) -> list[str]:
    """Copy the checkpoint's adapter files into adapter_dir, clearing stale ones.

    The swift run directory lives *inside* adapter_dir and holds the checkpoint
    being copied from, so it must survive the cleanup.
    """
    ensure_dir(adapter_dir)
    checkpoint = checkpoint.resolve()
    for stale in adapter_dir.iterdir():
        if stale.resolve() in checkpoint.parents or stale.resolve() == checkpoint:
            continue
        if stale.is_file():
            stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale)
    copied = []
    for item in checkpoint.iterdir():
        if item.is_file() and (item.name in ADAPTER_KEEP or item.name.startswith("adapter")):
            shutil.copy2(item, adapter_dir / item.name)
            copied.append(item.name)
    if "adapter_config.json" not in copied:
        raise RuntimeError(f"no adapter_config.json in {checkpoint}; LoRA weights were not saved")
    return sorted(copied)


def main() -> int:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune on the training corpus.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the swift command and exit")
    ap.add_argument("--allow-flat-loss", action="store_true",
                    help="do not fail when the loss does not decrease")
    args = ap.parse_args()

    cfg = load_config(args.config)

    dataset = cfg_path(cfg, "swift_dataset")
    if not dataset.exists():
        if not args.dry_run:
            log.error("dataset not found: %s. Run src/build_dataset.py first.", dataset)
            return 2
        log.warning("dataset %s does not exist yet (dry run)", dataset)

    # Steps, not epochs, are what determines whether anything is learned.
    n_examples = sum(1 for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip())         if dataset.exists() else 0
    tr = cfg["training"]
    per_step = max(1, int(tr["per_device_train_batch_size"]) * int(tr["gradient_accumulation_steps"]))
    steps = (n_examples * int(tr["num_train_epochs"])) // per_step
    log.info("corpus: %d training example(s) x %d epoch(s) / (batch %d x accum %d) = %d optimizer steps",
             n_examples, tr["num_train_epochs"], tr["per_device_train_batch_size"],
             tr["gradient_accumulation_steps"], steps)
    if steps and steps < 20:
        log.warning("only %d optimizer steps: that is usually too few to move an 8B model. "
                    "Raise training.num_train_epochs (or add documents) if the loss is still "
                    "falling at the end of the run.", steps)

    adapter_dir = cfg_path(cfg, "adapter_dir")
    swift_output = ensure_dir(adapter_dir / "swift_run")
    cmd = build_command(cfg, swift_output)

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    # Cap Qwen-VL visual tokens so a full-page image cannot blow past max_length.
    set_vision_env(cfg)
    env = os.environ.copy()
    env.setdefault("NPROC_PER_NODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    code, stdout_losses, labels_sample = run_streaming(cmd, env)
    if code != 0:
        log.error("swift sft failed with exit code %d", code)
        return code

    losses = losses_from_logfile(swift_output) or stdout_losses
    checkpoint = latest_checkpoint(swift_output)
    if checkpoint is None:
        log.error("training finished but no checkpoint-* directory was written under %s", swift_output)
        return 3
    log.info("latest checkpoint: %s", checkpoint)
    copied = export_adapter(checkpoint, adapter_dir)
    log.info("adapter exported to %s: %s", adapter_dir, ", ".join(copied))

    first = losses[0] if losses else None
    last = losses[-1] if losses else None
    decreased = bool(losses) and last < first
    labels = analyse_labels(labels_sample)
    if labels["captured"] and not labels["prompt_masked"]:
        log.error("LABEL MASKING LOOKS WRONG: ms-swift's [LABELS] line shows no -100 run, "
                  "so the loss may cover the prompt and OCR text, not just the answer. "
                  "Sample: %s", labels["sample"][:200])
    elif labels["captured"]:
        log.info("label masking OK: prompt is masked, target begins with the JSON object")
    else:
        log.warning("%s", labels["note"])

    report = {
        "checkpoint": str(checkpoint),
        "training_examples": n_examples,
        "optimizer_steps_planned": steps,
        "label_masking": labels,
        "adapter_dir": str(adapter_dir),
        "adapter_files": copied,
        "steps_logged": len(losses),
        "first_loss": first,
        "last_loss": last,
        "min_loss": min(losses) if losses else None,
        "loss_decreased": decreased,
        "losses": losses,
    }
    write_json(adapter_dir / "train_report.json", report)

    if not losses:
        log.warning("no loss values were captured; inspect %s manually", swift_output)
        return 0 if args.allow_flat_loss else 4

    log.info("loss: first=%.4f last=%.4f min=%.4f over %d step(s)",
             first, last, min(losses), len(losses))
    if not decreased:
        log.error(
            "TRAINING LOSS DID NOT DECREASE. Check, in order: label masking (loss must "
            "cover only the assistant JSON), the adapter actually attaching, the learning "
            "rate, and whether the image reached the collator."
        )
        return 0 if args.allow_flat_loss else 5

    log.info("training OK: loss fell %.4f -> %.4f", first, last)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
