"""Stage 4 - merge the LoRA adapter into the base model to produce v1.

The base is loaded in bf16 (NOT 4-bit: merging into quantized weights loses the
adapter's precision), the adapter is attached with PEFT, merge_and_unload folds
it in, and the result is saved to outputs/merged_v1/ together with the processor
so it can be loaded standalone.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cfg_path, ensure_dir, load_config, read_json, setup_logging, write_json  # noqa: E402
from modeling import load_model_and_processor  # noqa: E402

log = setup_logging("merge")


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge the trained adapter into the base model.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu",
                    help="device_map for the merge: cpu (safe, ~16GB RAM) or auto (GPU)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing merged_v1")
    args = ap.parse_args()

    cfg = load_config(args.config)
    adapter_dir = cfg_path(cfg, "adapter_dir")
    merged_dir = cfg_path(cfg, "merged_dir")

    if not (adapter_dir / "adapter_config.json").exists():
        log.error("no adapter_config.json in %s. Run src/train.py first.", adapter_dir)
        return 2

    if any(merged_dir.glob("*.safetensors")) and not args.force:
        log.error("%s already holds a merged model; pass --force to overwrite", merged_dir)
        return 2

    adapter_cfg = read_json(adapter_dir / "adapter_config.json")
    log.info(
        "adapter: r=%s alpha=%s targets=%s base=%s",
        adapter_cfg.get("r"), adapter_cfg.get("lora_alpha"),
        adapter_cfg.get("target_modules"), adapter_cfg.get("base_model_name_or_path"),
    )

    from peft import PeftModel

    base_id = cfg["model"]["base_model_id"]
    model, processor = load_model_and_processor(
        base_id, cfg, load_in_4bit=False, device_map=args.device,
        revision=cfg["model"].get("revision"),
    )

    log.info("attaching adapter from %s", adapter_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    trainable = [n for n, _ in model.named_parameters() if "lora_" in n]
    if not trainable:
        log.error("adapter attached but no lora_* parameters are present; the merge would be a no-op")
        return 3
    log.info("adapter attached: %d LoRA tensors", len(trainable))

    log.info("merging (merge_and_unload)")
    merged = model.merge_and_unload()

    ensure_dir(merged_dir)
    if args.force:
        for stale in merged_dir.iterdir():
            (shutil.rmtree if stale.is_dir() else Path.unlink)(stale)

    log.info("saving merged model to %s", merged_dir)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    processor.save_pretrained(str(merged_dir))

    shards = sorted(p.name for p in merged_dir.glob("*.safetensors"))
    if not shards:
        log.error("save produced no safetensors shards in %s", merged_dir)
        return 3
    size_gb = sum(p.stat().st_size for p in merged_dir.glob("*.safetensors")) / 1e9

    write_json(merged_dir / "merge_report.json", {
        "base_model_id": base_id,
        "adapter_dir": str(adapter_dir),
        "lora_tensor_count": len(trainable),
        "shards": shards,
        "total_gb": round(size_gb, 2),
    })
    log.info("merged v1 written: %d shard(s), %.2f GB", len(shards), size_gb)

    # Acceptance check: the merged directory must be loadable on its own.
    from transformers import AutoConfig

    AutoConfig.from_pretrained(str(merged_dir), trust_remote_code=True)
    if not (merged_dir / "preprocessor_config.json").exists():
        log.warning("no preprocessor_config.json saved; infer.py will fall back to the base processor")
    log.info("merged model config loads standalone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
