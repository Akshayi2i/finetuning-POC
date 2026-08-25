"""Single source of truth for the conversation shape.

build_dataset.py (training) and infer.py (inference) both build their messages
here, so the system prompt and the ocr_plus_image modality cannot drift apart.
Prompt/modality drift is one of the named failure points in the spec.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from common import cfg_path, ocr_dir_for, read_text, require_pages

IMAGE_TAG = "<image>"


def load_system_prompt(cfg: dict) -> str:
    return read_text(cfg_path(cfg, "system_prompt")).strip()


def build_ocr_text(pages: list[tuple[Path, Path]]) -> str:
    """Concatenate per-page OCR markdown with explicit page delimiters."""
    chunks = []
    for idx, (_, md) in enumerate(pages, start=1):
        chunks.append(f"--- OCR page {idx} ---\n{read_text(md).strip()}")
    return "\n\n".join(chunks)


def build_user_content(pages: list[tuple[Path, Path]], ocr_text: str) -> list[dict]:
    """User turn: every page image first, then the concatenated OCR markdown."""
    content: list[dict] = [{"type": "image", "image": str(p.as_posix())} for p, _ in pages]
    content.append({"type": "text", "text": ocr_text})
    return content


def build_messages(cfg: dict, doc_id: str,
                   assistant_text: str | None = None) -> tuple[list[dict], list[str]]:
    """Return (messages, image_paths) for one document, in the spec's chat format.

    assistant_text is the compact gold JSON for training, None for inference. Every
    document in the corpus is built by this one function, so training examples and
    the inference prompt cannot drift apart.
    """
    pages = require_pages(ocr_dir_for(cfg, doc_id))
    ocr_text = build_ocr_text(pages)
    messages: list[dict] = [
        {"role": "system", "content": load_system_prompt(cfg)},
        {"role": "user", "content": build_user_content(pages, ocr_text)},
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    return messages, [str(p.as_posix()) for p, _ in pages]


def to_swift_record(messages: list[dict], images: list[str]) -> dict[str, Any]:
    """Convert the spec record into ms-swift's native multimodal form.

    ms-swift expects flat string content with one <image> tag per image plus a
    top-level "images" list. Same conversation, same order, different encoding.
    """
    swift_messages = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            swift_messages.append({"role": msg["role"], "content": content})
            continue
        parts = []
        for block in content:
            if block.get("type") == "image":
                parts.append(IMAGE_TAG)
            elif block.get("type") == "text":
                parts.append(block["text"])
        swift_messages.append({"role": msg["role"], "content": "\n".join(parts)})
    return {"messages": swift_messages, "images": list(images)}


def conversation_fingerprint(messages: list[dict]) -> str:
    """Hash of the prompt + modality, minus the assistant turn.

    build_dataset.py stores it and infer.py re-checks it; a mismatch means the
    training and inference conversations diverged.

    Image blocks contribute their file *name* only, never the full path, so
    moving the project between machines (build locally, train on a rented GPU)
    does not read as prompt drift.
    """
    h = hashlib.sha256()
    for msg in messages:
        if msg["role"] == "assistant":
            continue
        h.update(msg["role"].encode("utf-8"))
        content = msg["content"]
        if isinstance(content, str):
            h.update(content.encode("utf-8"))
            continue
        for block in content:
            kind = block.get("type", "")
            h.update(kind.encode("utf-8"))
            if kind == "image":
                h.update(Path(str(block.get("image", ""))).name.encode("utf-8"))
            else:
                h.update(str(block.get("text", "")).encode("utf-8"))
    return h.hexdigest()[:16]
