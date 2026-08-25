"""No-GPU smoke test of everything except the three model stages.

Builds a synthetic corpus (one "real" document plus synthetic siblings) in a
throwaway directory and runs:

  make_sample_pdf -> run_ocr.py (--engine pymupdf) -> build_dataset.py
  -> fabricated base/v1 results -> compare.py

asserting the corpus pairing, the held-out split, the record shape, the ms-swift
encoding, the scoring, the confidence reporting and every verdict path. It proves
the plumbing before a GPU box is touched. It does not exercise train.py /
merge.py / infer.py, which need CUDA.

    python tests/smoke_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import yaml  # noqa: E402

from common import compact_json, read_json  # noqa: E402
from make_sample_pdf import GOLD, build_pdf, variant  # noqa: E402

PY = sys.executable
FAILURES: list[str] = []
CORPUS_SIZE = 4  # 1 test document + 3 synthetic siblings


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def run(label: str, *args: str) -> subprocess.CompletedProcess:
    print(f"\n--- {label}")
    proc = subprocess.run([PY, *args], capture_output=True, text=True, cwd=ROOT)
    for line in (proc.stdout + proc.stderr).strip().splitlines()[-12:]:
        print(f"    | {line}")
    return proc


def build_corpus(data_dir: Path) -> str:
    """One 'real' document plus synthetic siblings, filed like the user's data/."""
    pdfs = data_dir / "training sample"
    golds = data_dir / "golden json"
    pdfs.mkdir(parents=True, exist_ok=True)
    golds.mkdir(parents=True, exist_ok=True)

    build_pdf(pdfs / "Client Application 0.pdf", GOLD)
    (golds / "Client_Application_0_extraction.json").write_text(
        json.dumps(GOLD, indent=2), encoding="utf-8")
    for i in range(1, CORPUS_SIZE):
        gold = variant(i)
        build_pdf(pdfs / f"Client Application {i}.pdf", gold)
        (golds / f"Client_Application_{i}_extraction.json").write_text(
            json.dumps(gold, indent=2), encoding="utf-8")
    return "client_application_0"


def make_config(work: Path) -> Path:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["ocr"]["engine"] = "pymupdf"
    cfg["corpus"]["test_document"] = "client_application_0"
    # The repo's schema describes the real document; the synthetic sample gets its own.
    sample_schema = work / "schema.json"
    sample_schema.write_text(json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": list(GOLD),
        "additionalProperties": True,
    }, indent=2), encoding="utf-8")

    cfg["paths"].update({
        "data_dir": str(work / "data"),
        "system_prompt": str(ROOT / "prompts" / "system_prompt.txt"),
        "json_schema": str(sample_schema),
        "ocr_dir": str(work / "ocr"),
        "dataset": str(work / "dataset" / "train.jsonl"),
        "swift_dataset": str(work / "dataset" / "train_swift.jsonl"),
        "adapter_dir": str(work / "adapter"),
        "merged_dir": str(work / "merged_v1"),
        "results_dir": str(work / "results"),
    })
    cfg_file = work / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_file


def degraded_output() -> dict:
    """What an un-tuned model plausibly returns: right idea, wrong details."""
    out = json.loads(json.dumps(GOLD))
    out["policy_number"] = "GL 0042198"                 # reformatted identifier
    out["effective_date"] = "April 1, 2025"             # normalizer should still match this
    out["issue_date"] = None                            # missed
    out["taxes_and_fees"] = None                        # missed
    out["insured_address"] = "1450 Dockside Rd, Halifax NS"   # paraphrased
    out["coverages"] = out["coverages"][:2]             # dropped a row
    out["coverages"][1]["premium"] = 3105.0             # misread digit
    out["notes"] = None
    return out


def write_result(results: Path, name: str, payload: dict, **meta) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{name}_output.json").write_text(json.dumps(payload), encoding="utf-8")
    base = {"model": name, "json_valid_strict": True, "repaired_from_truncation": False,
            "hit_max_new_tokens": False}
    base.update(meta)
    (results / f"{name}_meta.json").write_text(json.dumps(base), encoding="utf-8")


def confidence(mean: float, tokens: int = 500, low: float = 0.1) -> dict:
    return {"tokens": tokens, "mean_token_probability": mean, "mean_token_logprob": -0.3,
            "min_token_probability": 0.01, "p10_token_probability": 0.4,
            "median_token_probability": mean, "low_confidence_tokens": int(tokens * low),
            "low_confidence_fraction": low}


def unit_checks() -> None:
    """Regression checks for pieces the staged run cannot reach."""
    from common import compact_json as cj
    from common import name_key, recover_json_object, slugify
    from compare import normalize
    from compare import score as score_fn
    from prompting import conversation_fingerprint
    from train import analyse_labels, export_adapter

    print("\n=== unit checks ===")

    # Corpus pairing must survive the naming styles a labelling process produces.
    for pdf_stem, gold_stem in [
        ("Signed Application - Client 6", "Signed_Application_Client_6_extraction"),
        ("synthetic_001", "synthetic_001_gold"),
        ("Doc-7 (copy)", "doc 7 (copy) golden"),
        ("client_9", "client_9_ground_truth"),
    ]:
        check(f"pairs {pdf_stem!r} with its gold", name_key(pdf_stem) == name_key(gold_stem))
    check("different documents do not collide", name_key("client_9") != name_key("client_10"))
    check("doc ids are filesystem safe", slugify("Signed Application - Client 6")
          == "signed_application_client_6")

    # The swift run directory lives inside adapter_dir; exporting must not delete it.
    with tempfile.TemporaryDirectory(prefix="poc_export_") as tmp:
        adapter = Path(tmp) / "adapter"
        ckpt = adapter / "swift_run" / "v0-run" / "checkpoint-15"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
        (ckpt / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
        (adapter / "stale.safetensors").write_text("old", encoding="utf-8")
        (adapter / "junk").mkdir()
        copied = export_adapter(ckpt, adapter)
        check("export_adapter copies the adapter files", copied == [
            "adapter_config.json", "adapter_model.safetensors"], str(copied))
        check("export_adapter keeps the swift run directory", ckpt.exists())
        check("export_adapter clears stale adapter files",
              not (adapter / "stale.safetensors").exists() and not (adapter / "junk").exists())

    # The fingerprint must survive moving the project to another machine.
    def msgs(prefix: str) -> list[dict]:
        return [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": [
                {"type": "image", "image": f"{prefix}/page_1.png"},
                {"type": "text", "text": "ocr"},
            ]},
        ]

    check("fingerprint ignores the image directory",
          conversation_fingerprint(msgs("/home/pod/outputs/ocr"))
          == conversation_fingerprint(msgs("D:/POC2/outputs/ocr")))
    check("fingerprint still reacts to prompt changes",
          conversation_fingerprint(msgs("a")) != conversation_fingerprint(
              [{"role": "system", "content": "different"}, msgs("a")[1]]))

    # Identifiers must not be coerced to numbers.
    check("identifier keeps its leading zeros", normalize("policy_number", "007") != normalize(
        "policy_number", 7), str(normalize("policy_number", "007")))
    check("money strings still normalize", normalize("total_premium", "$18,450.00") == 18450.0)

    # A model that hits max_new_tokens emits a valid prefix, not valid JSON.
    full = cj(GOLD)
    obj, _, repaired = recover_json_object(full)
    check("complete output parses without repair", obj == GOLD and not repaired)

    obj, reason, repaired = recover_json_object(full[: int(len(full) * 0.85)])
    check("truncated output is recovered", obj is not None, reason)
    check("truncated output is flagged as repaired", repaired)
    if obj is not None:
        partial = score_fn(GOLD, obj)
        check("repaired output earns partial credit, not zero",
              0 < partial["match_rate"] < 1.0, f"{partial['match_rate']:.2f}")

        def leaves(o, p=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield from leaves(v, f"{p}.{k}")
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    yield from leaves(v, f"{p}[{i}]")
            else:
                yield p, o

        gold_leaves = dict(leaves(GOLD))
        check("repair never invents a value",
              all(k in gold_leaves and gold_leaves[k] == v for k, v in leaves(obj)))

    check("garbage is not 'repaired' into an object",
          recover_json_object("I cannot read this document.")[0] is None)

    # Confidence summary maths.
    from infer import confidence_stats

    stats = confidence_stats([-0.01, -0.02, -3.0])
    check("confidence reports the geometric-mean token probability",
          0.3 < stats["mean_token_probability"] < 0.5, str(stats["mean_token_probability"]))
    check("confidence counts low-probability tokens", stats["low_confidence_tokens"] == 1)
    check("confidence handles an empty generation", confidence_stats([])["tokens"] == 0)

    # Completeness: coverage of the OCR text by the extracted output.
    from compare import coverage_report

    ocr_sample = "\n".join([
        "Policy Number: GL-0042198-01",
        "Total Premium: $18,450.00   Taxes and Fees: $1,476.00",
        "Commercial General Liability   5,000,000   2,500   12,300.00",
        "Named Insured: Harbour Point Logistics Inc.",
    ])
    full = coverage_report(ocr_sample, {"everything": ocr_sample})
    check("an output containing the whole page scores full coverage",
          full["number_coverage"] == 1.0 and full["word_coverage"] == 1.0, str(full))
    partial = coverage_report(ocr_sample, {"policy_number": "GL-0042198-01"})
    check("a partial extraction scores partial coverage",
          0 < partial["number_coverage"] < 1.0, str(partial["number_coverage"]))
    check("coverage names the numbers that were skipped",
          "18450.00" in partial["missing_numbers"] and "5000000" in partial["missing_numbers"],
          str(partial["missing_numbers"]))
    empty = coverage_report(ocr_sample, {})
    check("an empty output scores zero coverage", empty["number_coverage"] == 0.0)
    check("coverage is skipped when there is nothing to compare",
          coverage_report("", {"a": 1}) == {} and coverage_report(ocr_sample, None) == {})

    # The prompt is what the model is trained against: every section the schema
    # requires must be described in it, or the model is scored on fields it was
    # never asked for.
    import json as _json

    prompt_text = (ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
    schema = _json.loads((ROOT / "prompts" / "schema.json").read_text(encoding="utf-8"))
    unnamed = [s for s in schema["required"] if s not in prompt_text]
    check("every schema section is described in the extraction prompt",
          not unnamed, f"missing: {unnamed}")
    import re as _re

    listed = [m[1] for m in _re.findall(r'^\s*(\d+)\.\s+"([a-z_]+)"', prompt_text, _re.M)]
    required = schema["required"]
    check("the prompt lists the required sections in schema order",
          listed[:len(required)] == required,
          f"prompt: {listed[:3]}... schema: {required[:3]}...")
    extra_listed = listed[len(required):]
    check("any extra section the prompt describes is documented in the schema",
          all(name in schema["properties"] for name in extra_listed), str(extra_listed))
    check("the prompt offers a catch-all so unanticipated content is not dropped",
          "additional_content" in prompt_text and "additional_content" in schema["properties"])
    for phrase in ("COMPLETE extraction", "MINIMUM, not a limit", "account for every page"):
        check(f"prompt demands completeness: {phrase!r}", phrase in prompt_text)
    for rule in ("[NOT ASKED]", "null", "no markdown code fences", "verbatim"):
        check(f"prompt states the {rule!r} rule", rule in prompt_text)

    # ms-swift's [LABELS] line is the spec's label-masking check.
    check("masked [LABELS] line reads as correct",
          analyse_labels('[-100 * 22841]{"document_metadata":{"a":1}}')["prompt_masked"])
    check("unmasked [LABELS] line is caught",
          not analyse_labels("You are an insurance document extraction engine...")["prompt_masked"])
    check("absent [LABELS] line is reported, not assumed",
          analyse_labels(None)["captured"] is False)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="poc_smoke_") as tmp:
        work = Path(tmp)
        test_doc = build_corpus(work / "data")
        cfg_file = make_config(work)
        cfg = str(cfg_file)

        print(f"\n=== stage 1: OCR ({CORPUS_SIZE} documents) ===")
        proc = run("run_ocr", "src/run_ocr.py", "--config", cfg)
        check("run_ocr exits 0", proc.returncode == 0, proc.stderr[-400:])
        corpus_meta = work / "ocr" / "corpus_meta.json"
        check("corpus_meta.json written", corpus_meta.exists())
        if corpus_meta.exists():
            meta = read_json(corpus_meta)
            check(f"all {CORPUS_SIZE} documents OCR'd", len(meta["documents"]) == CORPUS_SIZE,
                  str(len(meta["documents"])))
            check("no document failed OCR", not meta["failed"], str(meta["failed"]))
        check("per-document OCR folders exist",
              (work / "ocr" / test_doc / "page_1.png").exists()
              and (work / "ocr" / test_doc / "page_1.md").exists())
        if (work / "ocr" / test_doc / "page_1.png").exists():
            from PIL import Image

            with Image.open(work / "ocr" / test_doc / "page_1.png") as img:
                cap = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))[
                    "model"]["max_image_long_side_px"]
                check("image respects max_image_long_side_px", max(img.size) <= cap, str(img.size))

        print("\n=== stage 2: training corpus ===")
        proc = run("build_dataset", "src/build_dataset.py", "--config", cfg, "--no-token-count")
        check("build_dataset exits 0", proc.returncode == 0, proc.stderr[-400:])
        train = work / "dataset" / "train.jsonl"
        check("train.jsonl written", train.exists())
        if train.exists():
            lines = [ln for ln in train.read_text(encoding="utf-8").splitlines() if ln.strip()]
            check("test document is held out of training",
                  len(lines) == CORPUS_SIZE - 1, f"{len(lines)} lines for {CORPUS_SIZE} docs")
            record = json.loads(lines[0])
            roles = [m["role"] for m in record["messages"]]
            check("roles are system/user/assistant", roles == ["system", "user", "assistant"],
                  str(roles))
            user = record["messages"][1]["content"]
            check("user turn has an image block", any(b["type"] == "image" for b in user))
            check("user turn has a text block", any(b["type"] == "text" for b in user))
            golds = {compact_json(variant(i)) for i in range(1, CORPUS_SIZE)}
            assistants = {json.loads(ln)["messages"][2]["content"] for ln in lines}
            check("each assistant turn is that document's own gold", assistants == golds)
            check("the test document's gold is absent from training",
                  compact_json(GOLD) not in assistants)
            swift = [json.loads(ln) for ln in
                     (work / "dataset" / "train_swift.jsonl").read_text(
                         encoding="utf-8").splitlines() if ln.strip()]
            check("swift file has one record per training document",
                  len(swift) == CORPUS_SIZE - 1)
            check("swift <image> tags match the images list",
                  all(r["messages"][1]["content"].count("<image>") == len(r["images"])
                      for r in swift))
            meta = read_json(work / "dataset" / "dataset_meta.json")
            check("dataset_meta records the test document", meta["test_document"] == test_doc)
            check("dataset_meta records the held-out split", meta["test_held_out"] is True)
            check("dataset_meta records every document", len(meta["documents"]) == CORPUS_SIZE)

        print("\n=== stage 2: guards ===")
        ocr_md = work / "ocr" / test_doc / "page_1.md"
        kept = ocr_md.read_text(encoding="utf-8")
        ocr_md.write_text("\n", encoding="utf-8")
        proc = run("build_dataset (one document with empty OCR)", "src/build_dataset.py",
                   "--config", cfg, "--no-token-count")
        check("a document with empty OCR is excluded, not trained image-only",
              "OCR text empty" in (proc.stdout + proc.stderr) or proc.returncode == 3)
        ocr_md.write_text(kept, encoding="utf-8")
        run("build_dataset (restored)", "src/build_dataset.py", "--config", cfg,
            "--no-token-count")

        from PIL import Image as PILImage

        from build_dataset import image_tokens

        probe = work / "probe_792x1024.png"
        PILImage.new("RGB", (792, 1024), "white").save(probe)
        check("visual-token formula matches Qwen3-VL (792x1024 -> 784)",
              image_tokens([str(probe)]) == 784, str(image_tokens([str(probe)])))
        check("visual tokens scale with page count",
              image_tokens([str(probe)] * 15) == 784 * 15)

        print("\n=== stage 7: compare (base worse, v1 perfect) ===")
        results = work / "results" / test_doc
        fp = read_json(work / "dataset" / "dataset_meta.json")["test_fingerprint"]
        write_result(results, "base", degraded_output(), conversation_fingerprint=fp,
                     confidence=confidence(0.62, low=0.18), weight_signature=1.0)
        write_result(results, "v1", GOLD, conversation_fingerprint=fp,
                     confidence=confidence(0.91, low=0.03), weight_signature=2.0)
        proc = run("compare", "src/compare.py", "--config", cfg, "--quiet")
        check("compare exits 0 (PASS verdict)", proc.returncode == 0, proc.stdout[-400:])
        comparison = work / "results" / test_doc / "comparison.json"
        check("comparison.json written", comparison.exists())
        if comparison.exists():
            report = read_json(comparison)
            check("verdict is PASS", report["verdict"] == "PASS", report["verdict"])
            check("report names the scored document", report["document"] == test_doc)
            check("report records the held-out split", report["test_held_out"] is True)
            check("report records the corpus size", report["corpus_size"] == CORPUS_SIZE)
            check("v1 scores 100% against gold", report["v1_match_rate"] == 1.0,
                  str(report["v1_match_rate"]))
            check("base scores below v1", report["base_match_rate"] < 1.0,
                  str(report["base_match_rate"]))
            # Only policy_number, insured_name, premiums and coverage premiums differ
            # between the synthetic variants; everything else is boilerplate.
            split = report.get("field_split") or {}
            check("corpus field variability is computed", bool(split), str(split)[:120])
            if split:
                # variant() changes 4 scalars + one premium on each of 3 coverages
                check("variable fields identified (the ones variant() changes)",
                      report["variable_fields"] == 7, str(report["variable_fields"]))
                check("boilerplate fields are the majority",
                      report["static_fields"] > report["variable_fields"],
                      f"{report['static_fields']} static vs {report['variable_fields']} variable")
                check("extraction (variable) rate is reported for both models",
                      split["base"]["variable"]["match_rate"] is not None
                      and split["v1"]["variable"]["match_rate"] is not None)
                check("base scores worse on the fields that vary",
                      split["base"]["variable"]["match_rate"]
                      < split["v1"]["variable"]["match_rate"],
                      f"{split['base']['variable']['match_rate']} vs "
                      f"{split['v1']['variable']['match_rate']}")
                check("the split is reported in the verdict reasons",
                      any("actually vary between documents" in r for r in report["reasons"]))
            check("confidence improvement is reported",
                  report["confidence"]["improved"] is True
                  and report["confidence"]["delta"] > 0, str(report.get("confidence")))
            fields = {f["field"]: f for f in report["per_field"]["base"]}
            check("date normalization matches 'April 1, 2025' to 2025-04-01",
                  fields["effective_date"]["match"])
            check("reformatted policy number is a miss", not fields["policy_number"]["match"])
            check("dropped coverage row is counted missing",
                  "coverages[2].premium" in report["missing_fields"]["base"])
            check("list row count recorded",
                  report["list_counts"]["base"]["coverages"] == {"gold": 3, "got": 2},
                  str(report["list_counts"]["base"].get("coverages")))

        print("\n=== compare: v1 no better than base ===")
        write_result(results, "v1", degraded_output(), conversation_fingerprint=fp,
                     confidence=confidence(0.62), weight_signature=1.0)
        proc = run("compare (equal outputs)", "src/compare.py", "--config", cfg, "--quiet")
        check("compare exits 1 on INVESTIGATE", proc.returncode == 1)
        report = read_json(comparison)
        check("verdict is INVESTIGATE", report["verdict"] == "INVESTIGATE", report["verdict"])
        check("identical-output hint is reported",
              any("identical match rates" in r for r in report["reasons"]), str(report["reasons"]))
        check("identical weight signatures are flagged",
              any("identical weight signatures" in r for r in report["reasons"]))

        print("\n=== compare: v1 hit max_new_tokens (repaired prefix) ===")
        from common import recover_json_object

        full = compact_json(GOLD)
        recovered, _, _ = recover_json_object(full[: int(len(full) * 0.9)])
        write_result(results, "v1", recovered, conversation_fingerprint=fp,
                     json_valid_strict=False, repaired_from_truncation=True,
                     hit_max_new_tokens=True, confidence=confidence(0.8), weight_signature=2.0)
        run("compare (repaired v1)", "src/compare.py", "--config", cfg, "--quiet")
        report = read_json(comparison)
        check("repaired v1 is scored, not zeroed", report["v1_match_rate"] > 0.5,
              str(report["v1_match_rate"]))
        check("repaired v1 still reports json_valid false",
              report["summary"]["v1"]["json_valid"] is False)
        check("repaired v1 warns that criterion 2 failed",
              any("NOT valid JSON" in r for r in report["reasons"]), str(report["reasons"]))

        print("\n=== compare: a result left over from an earlier run ===")
        write_result(results, "v1", GOLD, conversation_fingerprint="fingerprint_from_an_old_run",
                     confidence=confidence(0.9), weight_signature=2.0)
        run("compare (stale v1)", "src/compare.py", "--config", cfg, "--quiet")
        report = read_json(comparison)
        check("stale result is flagged, not silently scored",
              any("STALE" in r for r in report["reasons"]), str(report["reasons"]))

        print("\n=== compare: v1 emitted unparseable output ===")
        (results / "v1_output.json").unlink()
        (results / "v1_meta.json").unlink()
        (results / "v1_raw.txt").write_text("Sure! Here is the JSON: {oops", encoding="utf-8")
        run("compare (invalid v1)", "src/compare.py", "--config", cfg, "--quiet")
        report = read_json(comparison)
        check("invalid v1 scores 0", report["v1_match_rate"] == 0.0)
        check("invalid v1 flagged", any("nothing scoreable" in r for r in report["reasons"]),
              str(report["reasons"]))

    unit_checks()

    print("\n==============================================")
    if FAILURES:
        print(f"SMOKE TEST FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST PASSED: all checks green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
