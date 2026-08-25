"""Model/processor loading shared by merge.py and infer.py."""

from __future__ import annotations

import importlib.util
import logging

log = logging.getLogger("modeling")


def resolve_attn_impl(requested: str | None) -> str | None:
    """Fall back to sdpa when flash-attn is requested but not installed."""
    if not requested:
        return None
    if requested.startswith("flash") and importlib.util.find_spec("flash_attn") is None:
        log.warning("flash-attn not installed; falling back to attn_implementation='sdpa'")
        return "sdpa"
    return requested


def model_class():
    """Prefer the concrete Qwen3-VL class, fall back to the auto class."""
    import transformers

    for name in ("Qwen3VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            log.info("using model class %s", name)
            return cls
    from transformers import AutoModelForImageTextToText

    log.info("using model class AutoModelForImageTextToText")
    return AutoModelForImageTextToText


def bnb_config(cfg: dict):
    import torch
    from transformers import BitsAndBytesConfig

    q = cfg["qlora"]
    dtype = getattr(torch, q.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
    )


def load_model_and_processor(
    model_path: str,
    cfg: dict,
    *,
    load_in_4bit: bool = False,
    device_map: str = "auto",
    revision: str | None = None,
):
    import torch
    from transformers import AutoProcessor

    kwargs: dict = {
        "dtype": torch.bfloat16,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if revision:
        kwargs["revision"] = revision
    attn = resolve_attn_impl(cfg["model"].get("attn_implementation"))
    if attn:
        kwargs["attn_implementation"] = attn
    if load_in_4bit:
        kwargs["quantization_config"] = bnb_config(cfg)

    cls = model_class()
    log.info("loading %s (4bit=%s, device_map=%s, attn=%s)", model_path, load_in_4bit, device_map, attn)
    try:
        model = cls.from_pretrained(model_path, **kwargs)
    except TypeError:
        # transformers < 4.56 names the dtype argument torch_dtype
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = cls.from_pretrained(model_path, **kwargs)

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, **({"revision": revision} if revision else {})
    )
    return model, processor
