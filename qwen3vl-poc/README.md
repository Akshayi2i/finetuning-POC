# Qwen3-VL-8B-Instruct — fine-tuning verification POC

Does our fine-tuning pipeline actually work? This answers that, end to end and cheaply.

Extract a test document with the **base** `Qwen3-VL-8B-Instruct` and keep the result.
Fine-tune the same model with QLoRA on a corpus of documents of that type. Merge the
adapter into the base to form **v1**. Extract the *same* test document again with v1,
through the identical routine. Compare both against the hand-written gold JSON.

**PASS** means v1 matches the gold better than the base model did — the training loop
demonstrably changed the model in the intended direction.

## Pipeline

```
data/**.pdf + data/**.json      corpus: every PDF paired with its gold JSON
   └─ run_ocr.py               MinerU -> outputs/ocr/<doc_id>/page_{n}.png + .md
       └─ build_dataset.py     one chat example per document, test document held out
                               -> outputs/dataset/train.jsonl (+ train_swift.jsonl)
           ├─ infer.py --model base   BEFORE training -> results/<doc>/base_output.json
           ├─ train.py         ms-swift QLoRA over the corpus -> outputs/adapter/
           │   └─ merge.py     PEFT merge_and_unload -> outputs/merged_v1/
           │       └─ infer.py --model v1 -> results/<doc>/v1_output.json
           └─ compare.py       -> results/<doc>/comparison.json + verdict
```

The base extraction runs **before** training, so the "before" measurement is on disk
no matter what happens during the training run.

## Your data

Drop PDFs and their gold JSONs anywhere under `data/` — subfolders and naming are up to
you. Each PDF is paired with its gold JSON by **file name**, ignoring case, punctuation
and a trailing `_extraction` / `_gold` / `_ground_truth` / similar:

```
data/training sample/Signed Application - Client 6.pdf
data/golden json/Signed_Application_Client_6_extraction.json     ✓ pairs
```

Add synthetic documents by dropping in the PDF and its gold JSON. Nothing in the config
needs changing — `run_ocr.py` reports every pair it found, every PDF missing a gold, and
every gold missing a PDF.

### Which document is the test document

`corpus.test_document` in `config.yaml` (default: the alphabetically first). By default it
is **held out** of training, which is what makes the result meaningful:

| `include_test_in_training` | What a gain proves |
| --- | --- |
| `false` (default) | The model learned the document **type**. Real evidence, and the harder test — it needs enough training documents to generalise from. |
| `true` | The model memorised this document. A guaranteed signal if you only want to confirm the plumbing works, but weak evidence about extraction quality. |

Start with `false`. If the loss falls but v1 shows no gain, flip it to `true` for one run:
if the gain appears, training works and you simply need more or more varied documents.

## Setup

There are **two environments**, and mixing them up wastes an afternoon.

### Workstation (Windows or Linux) — generate data, run the checks

No CUDA, no compiler, no model weights. This is all a laptop needs:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements-tools.txt
```

The synthetic-data generator also needs the native GTK3 runtime for WeasyPrint, which
pip cannot supply:

```powershell
winget install --id GtkD.GtkPlusRuntime.x64 --source winget    # then open a NEW terminal
# Linux: sudo apt install -y libpango-1.0-0 libpangoft2-1.0-0 libcairo2
```

Verify: `python tests/smoke_test.py` and
`python -c "from weasyprint import HTML; print('ok')"`.

### GPU box (Linux, 80 GB A100/H100) — OCR, training, inference

**Do not install `requirements.txt` on a Windows workstation.** ms-swift pulls
`stringzilla`, which has no Windows wheel and needs the MSVC C++ toolchain, and an 8B
model cannot train on a workstation GPU regardless. The 15-page document is a ~33k-token
example, and at that length the logits dominate memory (see Memory below).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# optional, ~10 min build, falls back to sdpa if absent:
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Pin `model.revision` in `config.yaml` to a specific Qwen3-VL commit before the real run,
and confirm the installed ms-swift supports that revision.

**MinerU is mandatory if your PDFs are scans.** The reference document reports zero
characters of embedded text on all 15 pages, so `--engine pymupdf` yields empty OCR and
the model would be left with images only. `build_dataset.py` refuses to build an
image-only example rather than silently changing the modality. Confirm `mineru --version`
works before starting.

## Auditing the golden JSON

The gold is the ground truth: an error in it is trained in and then scored as correct, so
it is worth verifying independently rather than trusting whatever produced it.

```bash
python src/verify_gold.py              # structural checks, no OCR needed
python src/verify_gold.py --ocr        # also check every value against its page
python src/verify_gold.py --json report.json
```

Per document it re-derives, from the files on disk:

| Check | What it catches |
| --- | --- |
| `pages` | `total_pages` disagreeing with the PDF's real page count |
| `arithmetic` | premium columns that do not sum to their totals; parts that do not sum to the term amount |
| `dates` | unparseable dates, transaction after effective, expiry not one year on |
| `formats` | VIN length, ZIP digits, state codes, malformed money |
| `emptiness` | a field blank here that **every** other document fills |

and across the corpus: an identical key skeleton, no duplicate documents, and a count of
how many fields actually vary (a corpus where little varies teaches little).

With `--ocr` it also reads each PDF through the OCR stage and checks both directions —
every value in the gold appears on the page (nothing invented) and the page holds nothing
the gold omits (nothing missed).

It is deliberately independent of the generator, which validates its own output against a
page it rendered from that same record — circular, and blind to its own bugs. Exit code is
1 when anything fails, so it gates a pipeline run. Verified by injecting faults: a wrong
page count, a broken column sum, a truncated VIN, a shifted expiry date and a blanked
field were each caught with a precise message.

**It never reports "grounded" unless grounding actually ran** — `--ocr` without OCR output
fails rather than passing quietly.

## Sizing, measured not guessed

For the reference 15-page document:

| Component | Tokens |
| --- | --- |
| System prompt | 892 |
| 15 page images at `max_image_long_side_px: 1024` (792×1024 → 784 each) | 11,760 |
| OCR markdown, 15 pages | ~8,000–11,000 |
| **Gold answer** | **11,594** |
| **Training example** | **~32,500** |

`max_seq_length` is 40960 and `max_new_tokens` is 16384 for that reason: the stock 8192 /
2048 would truncate the answer to a fifth of itself. `build_dataset.py` prints every
document's token count and flags any that would be dropped — check it before training.

At 1024 px the small print stays legible (the DocuSign envelope ID in each page header
reads cleanly), so the images can verify the OCR rather than just decorate it.

### Memory

The dominant cost is not the 4-bit weights (~5.5 GB) but the logits: 33k tokens × 151,936
vocab is ~10 GB in bf16, and the fp32 loss upcast plus its gradient multiply that. If the
run OOMs, in order:

1. `training.use_liger_kernel: true` (fused cross-entropy — needs `pip install
   liger-kernel`; verify it supports Qwen3-VL in your ms-swift build).
2. `model.max_image_long_side_px: 896` — saves ~2,700 tokens across 15 pages.
3. `training.max_seq_length: 32768`, only after `build_dataset.py` confirms every
   example still fits.

## Run

```bash
./run_all.sh                 # full pipeline, prints the verdict
./run_all.ps1                # Windows equivalent
```

Or stage by stage:

```bash
python src/run_ocr.py                  # every document; --doc <id> for one, --skip-existing
python src/verify_gold.py --ocr        # audit the ground truth before training on it
python src/build_dataset.py            # per-document token counts, held-out split
python src/infer.py --model base       # the "before" measurement
python src/train.py                    # --dry-run prints the swift command only
python src/merge.py --device cpu       # --device auto to merge on the GPU
python src/infer.py --model v1         # the "after" measurement
python src/compare.py                  # --all-fields, --max-rows N, --doc <id>
```

`train.py` prints the optimizer-step count (`documents × epochs ÷ batch ÷ accum`) and
warns when it is too small to move an 8B model. Steps, not epochs, decide whether
anything is learned — raise `num_train_epochs` if the loss is still falling at the end.

Exit codes are meaningful: `train.py` exits 5 when the loss did not fall, `infer.py`
exits 4 when the output is not valid JSON, `compare.py` exits 1 on `INVESTIGATE`.
`run_all.sh` stops at the first failure in OCR, dataset, training or merge, but treats
the two inference steps as non-fatal — an unparseable output is itself a result, so
`compare.py` still runs and records it.

## Verifying without a GPU

```bash
python tests/smoke_test.py
```

92 checks. Builds a synthetic 4-document corpus in a temp directory and runs OCR →
dataset → compare, asserting the corpus pairing, the held-out split, the record shape,
the ms-swift encoding, date/currency normalization, truncation recovery, confidence
reporting, the extraction/boilerplate split, OCR coverage, prompt-to-schema consistency,
the label-masking analyser, adapter export, and every verdict path. It does
not exercise train.py / merge.py / infer.py, which need CUDA.

## Reading the result

`outputs/results/<doc_id>/comparison.json` holds per-field results for both models, both
match rates, the per-section breakdown, the confidence delta and the verdict. With
hundreds of fields the terminal prints the section table plus only the rows where base
and v1 disagree; `--all-fields` and `--max-rows N` widen that.

Five things to check, in order:

1. **Loss falls across steps** — `outputs/adapter/train_report.json` (`first_loss`,
   `last_loss`, `loss_decreased`, `optimizer_steps_planned`). If it does not fall, the
   training loop is misconfigured. The same report carries `label_masking`, captured from
   ms-swift's own `[LABELS]` line: `prompt_masked` must be true, or the loss is covering
   the prompt and OCR text instead of only the answer.
2. **Both models emit parseable JSON** — `<doc>/{base,v1}_meta.json` `json_valid_strict`.
   A long answer can run into `max_new_tokens` and emit a valid *prefix*. That is not
   valid JSON and is reported as such, but the prefix is still scored so the comparison
   stays informative — `repaired_from_truncation` marks it and the match rate is then a
   floor, not a measurement.
3. **v1 match rate > base match rate** — the pass condition. Read the **split** underneath
   it before you believe the headline:

   ```
   base match rate : 41.2%  (281/683 fields)
   v1   match rate : 88.0%  (601/683 fields)
     variable fields  (extraction ): 23.1% -> 71.4%   <- the number that matters
     static fields    (boilerplate): 58.9% -> 99.2%
   ```

   A form package is largely boilerplate — legal notices, disclosures, the printed
   question text — identical in every document of the type. A model learns those by
   rote, which lifts the overall rate without any gain in extraction skill. `compare.py`
   works out which fields actually differ across your corpus and scores them separately.
   **The variable-field rate is the real extraction measurement.** (Needs at least two
   gold files; with one document everything looks static and the split is skipped.)
4. **Nothing was skipped** — `comparison.json` → `coverage`. This compares the extracted
   output against the **OCR text of the document**, not against the gold, so it catches
   content that the gold and the model both ignore:

   ```
   OCR coverage    : numbers 71.4% -> 96.2%  (gold 98.1%)
                     words   64.0% -> 89.7%  (gold 92.3%)
   ```

   `missing_numbers` lists the specific figures that never reached the output — the
   fastest way to see what an extraction dropped. The gold's own coverage is reported
   alongside, which audits **your gold file** for completeness too: if the gold covers
   only 60% of the numbers on the page, no amount of training will produce a complete
   extraction, because the target itself is incomplete.

   Caveats, and they matter: OCR noise creates tokens nothing can match, so 100% is not
   reachable; and presence is not correctness — a value in the wrong field still counts as
   covered. Read the base → v1 delta rather than the absolute number.
5. **Confidence rose** — `comparison.json` → `confidence`. This is the model's own mean
   probability for the tokens it chose (greedy decoding, so it is the probability of the
   argmax). It is a second signal, independent of the field match rate: a model that has
   learned the target format is less uncertain about producing it. Treat a rise as
   corroboration, not proof — confidence can rise while accuracy does not.

Expect the base model to do reasonably on fields it can read straight off the page
(policy numbers, addresses, premiums) and badly on the conventions of your gold: the
`"[NOT ASKED]"` marker, the exact section nesting, `data_quality_notes`. Those are what
v1 should learn, and where the section table should light up.

### If base and v1 look identical

| Cause | Check |
| --- | --- |
| Inference loaded the base weights as v1 | `weight_signature` in `base_meta.json` vs `v1_meta.json` — they must differ. `compare.py` warns when they match. |
| Prompt or modality drift | `conversation_fingerprint`, recorded per document at build time and re-checked by `infer.py`, which aborts on a mismatch. `compare.py` also flags a result left over from an earlier run. |
| Too little training | `train_report.json` — the step count and the loss curve. Raise `num_train_epochs`, add documents, or set `include_test_in_training: true` to separate "not learning" from "not generalising". |

## Design notes

- **`src/prompting.py` is the single source of truth for the conversation.** Every
  training example and both inference runs are built by the same function, which is what
  keeps the `ocr_plus_image` modality and the system prompt identical everywhere.
- **Two dataset encodings, one corpus.** `train.jsonl` is the chat format (content
  blocks); `train_swift.jsonl` is the same conversations in ms-swift's native form
  (`<image>` tags plus a top-level `images` list), which is what `swift sft` consumes.
- **Merging happens in bf16, never 4-bit** — folding an adapter into quantized weights
  would throw away the precision it was trained at.
- **`MAX_PIXELS` is set from `max_image_long_side_px`** in every stage, so a page is
  tokenized identically during training and inference.
- **Over-long examples are dropped, not truncated** (`--truncation_strategy delete`): a
  right-truncated answer trains on a broken target and looks exactly like broken label
  masking.

## Out of scope

Gold JSON creation routine, foundation + per-type adapter hierarchy, confidence
*calibration* (this POC reports raw model confidence, it does not calibrate it),
classifier, page routing, quantization, Azure Blob, RunPod orchestration, model registry,
multiple document types.
