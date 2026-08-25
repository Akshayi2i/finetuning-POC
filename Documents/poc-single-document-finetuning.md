# POC Build Spec: Qwen3-VL-8B-Instruct Fine-Tuning Verification

> **Revised.** The original spec trained on a single document. It now trains on a corpus
> of one document type (the source document plus synthetic variants of it), holds one
> document out as the test document, and tracks extraction confidence alongside accuracy.
> The architecture is unchanged; the dataset and the measurement are stronger.

## Objective

Verify that the fine-tuning pipeline works end to end and that the model measurably improves. Extract a test document with base `Qwen3-VL-8B-Instruct` and keep the result. Fine-tune the same model with QLoRA on a corpus of documents of that type. Merge the adapter into the base to form `v1`. Run the same test document through the identical extraction routine with `v1`. Compare both outputs against the manually provided gold JSON. Success means `v1` moves measurably closer to the gold JSON than the base model, on both field accuracy and the model's own confidence, confirming that training changes model behavior in the intended direction.

This POC uses the same architecture as the full system (QLoRA, ms-swift, MinerU OCR, chat-format data, merge, inference) and replaces the labeling routine with manually supplied gold JSON.

## Scope

In scope:
- MinerU OCR of every PDF in the corpus.
- One chat-format training example per document, built from its page images, OCR text, and gold JSON.
- A held-out test document, excluded from training by default.
- QLoRA fine-tuning with ms-swift over the corpus.
- Merge adapter into base to produce `v1`.
- Inference on base and `v1` with the same test document and the identical routine.
- Comparison of both outputs against gold JSON: field accuracy, per-section breakdown, and token-level confidence.

Out of scope (deliberately excluded for the POC):
- Gold JSON creation routine (provided manually, including the synthetic variants).
- Foundation plus per-type adapter hierarchy (single adapter only).
- Confidence *calibration* — raw model confidence is reported, not calibrated.
- Classifier, page routing, quantization, Azure Blob, RunPod orchestration, registry.
- Multiple document types (the corpus is variants of one type).

## Expected Outcome and How to Read It

The pass condition is that `v1` matches the gold JSON better than the base model on the same test document. If base and `v1` produce identical output, training did not take effect and the pipeline has a defect (commonly label masking, adapter not loaded, or learning rate too low).

Two configurations, and what each result means:

- **Test document held out** (`corpus.include_test_in_training: false`, the default). A gain is evidence the model learned the document *type*. This is the meaningful measurement, and the harder one: it needs enough training documents to generalize from.
- **Test document included** (`true`). A gain only proves memorization. Use it to separate "training is broken" from "the corpus is too small": if the held-out run shows no gain but the included run does, the pipeline works and the corpus needs more or more varied documents.

Confidence is reported as a second, independent signal: the model's mean probability for the tokens it chose under greedy decoding. A model that has learned the target format is less uncertain producing it. Treat a rise as corroboration, not proof — confidence can rise while accuracy does not.

---

## Project Structure

```
qwen3vl-poc/
├── README.md
├── requirements.txt
├── config.yaml                     # model id, LoRA config, hyperparameters, paths
├── data/                           # provided: any layout, PDFs paired to golds by name
│   ├── training sample/*.pdf       # source document + synthetic variants
│   └── golden json/*.json          # one gold JSON per PDF
├── prompts/
│   ├── system_prompt.txt           # extraction instructions + section outline
│   └── schema.json                 # shallow contract validating gold and outputs
├── src/
│   ├── common.py                   # config, corpus discovery/pairing, JSON recovery
│   ├── prompting.py                # THE conversation builder (train and infer share it)
│   ├── modeling.py                 # model/processor loading
│   ├── run_ocr.py                  # MinerU: every PDF -> per-page markdown + image
│   ├── build_dataset.py            # one chat example per document, test doc held out
│   ├── train.py                    # QLoRA fine-tuning via ms-swift
│   ├── merge.py                    # merge adapter into base -> v1
│   ├── infer.py                    # extraction with a chosen model, records confidence
│   └── compare.py                  # score base vs v1 against the test document's gold
├── outputs/
│   ├── ocr/<doc_id>/               # page_1.png, page_1.md, ocr_meta.json
│   ├── dataset/train.jsonl         # one line per training document
│   ├── adapter/                    # trained LoRA adapter + train_report.json
│   ├── merged_v1/                  # merged bf16 model
│   └── results/<doc_id>/
│       ├── base_output.json
│       ├── v1_output.json
│       └── comparison.json
├── tests/smoke_test.py             # 70 no-GPU checks over a synthetic corpus
└── run_all.sh                      # runs the full POC end to end
```

---

## Configuration (`config.yaml`)

```yaml
model:
  base_model_id: Qwen/Qwen3-VL-8B-Instruct
  revision: null                    # pin at implementation time
  attn_implementation: flash_attention_2
  max_image_long_side_px: 1024        # 15 pages x 784 visual tokens each

qlora:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  # projector included via ms-swift multimodal target handling; ViT frozen
  train_vit: false

corpus:
  test_document: null               # doc_id or file stem; null = first document
  include_test_in_training: false   # false = held out (real evidence of learning)
  min_train_documents: 1

training:
  # Steps = documents x epochs / (batch x accum). Steps, not epochs, decide whether
  # anything is learned; train.py prints the count and warns when it is too small.
  num_train_epochs: 8
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  weight_decay: 0.0
  max_grad_norm: 1.0
  gradient_checkpointing: true
  bf16: true
  logging_steps: 1
  save_strategy: epoch
  seed: 42
  use_liger_kernel: false           # fused cross-entropy; first lever if the run OOMs
  max_seq_length: 40960             # the reference gold answer alone is 11,594 tokens

paths:
  data_dir: data                    # PDFs + gold JSONs, any layout, paired by file name
  system_prompt: prompts/system_prompt.txt
  json_schema: prompts/schema.json
  ocr_dir: outputs/ocr
  dataset: outputs/dataset/train.jsonl
  adapter_dir: outputs/adapter
  merged_dir: outputs/merged_v1
  results_dir: outputs/results

inference:
  mode: ocr_plus_image              # match the training modality
  temperature: 0.0                  # deterministic for fair comparison
  max_new_tokens: 16384             # must exceed the gold answer length
```

---

## Module Specifications

### `src/run_ocr.py`
- Input: every PDF under `paths.data_dir`, paired with its gold JSON by normalized file name.
- Run MinerU to produce per-page markdown; render each page to PNG at `max_image_long_side_px`.
- Write `outputs/ocr/<doc_id>/page_{n}.png` and `page_{n}.md`, plus a corpus manifest.
- Report every PDF with no gold and every gold with no PDF.
- Keep the MinerU call swappable (subprocess or library). Log page count.
- Acceptance: PNG and markdown exist for every page; resolution cap applied.

### Extraction completeness

The engine performs **complete extraction**: every piece of information present in the PDF must appear in the output - all key-value pairs, question/answer rows, table cells, headings, notices, disclosures, footnotes, signature blocks, stamps, form codes and page footers. The section structure is a floor, not a ceiling: content that does not fit a defined section goes into extra keys or the `additional_content` catch-all (page / label / content) rather than being dropped. Skipping content that is on the page is the primary extraction failure.

`compare.py` measures this directly, against the document rather than the gold: it compares the word and number tokens MinerU read off the pages against the tokens present in the extracted JSON, reports the coverage for base and v1, and names the specific numbers that never reached the output. The gold's own coverage is reported alongside, so an incomplete gold is visible rather than assumed correct.

### `prompts/system_prompt.txt`
- Extraction instructions plus the target JSON schema for this document type.
- Rules: output only JSON, no code fences; missing fields become null; dates in YYYY-MM-DD; use OCR text as primary source and the image to verify and correct.
- The same prompt is used in training and inference (no drift).

### `src/verify_gold.py`
- Audits the golden JSON corpus independently of whatever produced it, because a generator that validates its own output against a page it rendered from that same record cannot see its own bugs.
- Per document: `total_pages` against the PDF's real page count; premium columns against their totals and the parts against the term amount; date parsing and ordering; VIN/ZIP/state/money formats; and any field left blank here that every other document fills.
- Across the corpus: identical key skeleton, no duplicate documents, and how many fields actually vary.
- With `--ocr`: every value in the gold must appear on its page (nothing invented) and the page must hold nothing the gold omits (nothing missed). Never reports "grounded" unless grounding actually ran.
- Exit code 1 on any failure, so it gates the pipeline. Runs after OCR and before dataset assembly.

### `src/build_dataset.py`
- Assemble one chat-format record **per document** and write `outputs/dataset/train.jsonl`, one line each. The test document (`corpus.test_document`) is held out unless `corpus.include_test_in_training` is true.
- Record shape:
```json
{
  "messages": [
    {"role": "system", "content": "<contents of system_prompt.txt>"},
    {"role": "user", "content": [
      {"type": "image", "image": "outputs/ocr/<doc_id>/page_1.png"},
      {"type": "text", "text": "<this document's OCR markdown>"}
    ]},
    {"role": "assistant", "content": "<this document's gold JSON, compact>"}
  ]
}
```
- If the PDF has multiple pages, include multiple image blocks and concatenate the OCR markdown.
- Validate that each gold parses and matches the schema. Log every document's token count so `max_seq_length` can be confirmed, and refuse any document whose OCR produced no text (the example would silently become image-only).
- An unusable *test* document is fatal: the whole comparison would otherwise rest on a broken input.
- Acceptance: one valid JSONL line per training document; each assistant turn equals that document's gold JSON; the test document's gold is absent when held out.

### `src/train.py`
- Load base model with the QLoRA 4-bit config; apply LoRA per `config.yaml`; freeze the ViT.
- Train with ms-swift on the corpus JSONL. ms-swift provides the multimodal collator and `-100` label masking (loss only on the assistant JSON).
- Save the adapter to `outputs/adapter/`.
- Log the training loss every step and the planned optimizer-step count (`documents x epochs / (batch x accum)`); steps, not epochs, determine whether anything is learned.
- Capture ms-swift's `[LABELS]` line for the first sample and assert the prompt is masked - the label-masking check below, performed rather than assumed.
- Acceptance: adapter saved; training loss decreases markedly from first to last step.

### `src/merge.py`
- Load base model plus the trained adapter; run PEFT `merge_and_unload`; save the merged model to `outputs/merged_v1/` in bf16.
- Acceptance: merged model loads standalone without the adapter.

### `src/infer.py`
- Arguments: `--model {base|v1}`, `--doc <doc_id>` (defaults to the test document); reuses the same prompt, images, and OCR text as training (mode `ocr_plus_image`, temperature 0).
- `base` loads `base_model_id`; `v1` loads `outputs/merged_v1/`.
- Run generation, parse the output as JSON, write `outputs/results/<doc_id>/{model}_output.json`.
- Record per-token confidence (log-probability of each chosen token under greedy decoding) and a weight signature, so a v1 run that silently loaded the base weights is detectable.
- An output that hits `max_new_tokens` is a valid prefix, not valid JSON: recover the prefix for scoring but keep reporting it as invalid.
- Acceptance: both models produce parseable JSON for the same input.

### `src/compare.py`
- Load the test document's gold JSON, `base_output.json`, `v1_output.json` and both result metas.
- Compute, for base and v1 against gold:
  - JSON validity (parses strictly, matches expected keys).
  - Field-level exact match rate, with light normalization for dates, currency, and casing.
  - Per-top-level-section match rates, so it is visible *where* the model improved.
  - A split between fields that vary across the corpus (the real extraction work) and fields identical in every document (boilerplate the model learns by rote). The variable-field rate is the extraction measurement; the overall rate is inflated by boilerplate.
  - List-field row count versus gold.
  - Confidence delta: mean token probability, base vs v1.
  - OCR coverage: the share of word and number tokens read off the pages that appear in the output, for base, v1 and the gold itself, with the missing numbers named.
- Write `outputs/results/<doc_id>/comparison.json` with per-field results, both match rates, the section breakdown and the confidence delta; print a summary table plus a verdict line: `PASS` if v1 match rate exceeds base match rate, else `INVESTIGATE`.
- Flag a result left over from an earlier run (fingerprint mismatch) rather than scoring it as current.
- Acceptance: comparison runs and emits a clear verdict.

### `run_all.sh`
- Sequentially: `run_ocr.py` → `build_dataset.py` → **`infer.py --model base`** → `train.py` → `merge.py` → `infer.py --model v1` → `compare.py`.
- The base extraction comes before training so the "before" measurement exists on disk regardless of what happens during the training run.
- Print the final verdict at the end.

---

## Environment

- Single GPU, 80 GB (A100/H100) for the reference 15-page document: each example is ~32,500 tokens, and at that length the logits (33k x 151,936 vocab, ~10 GB in bf16 before the fp32 loss upcast) dominate memory, not the 4-bit weights. Shorter documents fit smaller cards.
- MinerU is mandatory for scanned PDFs: they carry no text layer, so the pymupdf fallback yields nothing and the example would silently become image-only.
- `requirements.txt`: torch, transformers, ms-swift, peft, bitsandbytes, accelerate, flash-attn, MinerU, pillow, pyyaml, jsonschema. Pin versions at implementation time; verify ms-swift supports the target Qwen3-VL revision before running.

## Success Criteria

1. Training loss decreases across steps (learning is occurring), with the prompt confirmed masked.
2. Both base and `v1` produce parseable JSON for the test document.
3. `v1` field-match rate against gold exceeds the base field-match rate.
4. `v1` mean token confidence exceeds the base model's - corroboration, not proof.
5. The gain holds on the **variable** fields, not only on boilerplate identical across the corpus. The per-section breakdown shows where the document's conventions were learned.

If criterion 1 fails, the training loop is misconfigured (check label masking, learning rate, adapter attach). If criterion 3 fails while loss decreased, check in order: that inference loaded the merged `v1` rather than the base (compare `weight_signature` between the two result metas), that the same prompt and modality were used in training and inference (`conversation_fingerprint`), and whether the corpus is large enough for the held-out test to improve.

## Common Failure Points to Check First

- Label masking: loss must be computed only on assistant JSON tokens. `train.py` captures ms-swift's `[LABELS]` line for the first sample and asserts the prompt is masked.
- Inference loading the wrong weights: confirm `v1` path resolves to the merged model, not the base.
- Prompt or modality drift: training and inference must use the identical system prompt and `ocr_plus_image` mode.
- Too few optimizer steps: `documents x epochs / (batch x accum)`. A handful of steps will not move an 8B model; raise `num_train_epochs` until the loss clearly falls.
- No gain on a held-out test document while the loss falls: training works but the corpus is too small or too uniform to generalise from. Set `include_test_in_training: true` for one run to confirm the pipeline, then add documents.
- Multimodal collation: confirm the image is actually passed and tokenized, not dropped.
