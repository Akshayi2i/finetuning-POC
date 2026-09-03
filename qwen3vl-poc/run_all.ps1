# End-to-end POC run for Windows hosts. Mirrors run_all.sh.
# Usage: .\run_all.ps1 [-SkipTrain] [-Engine mineru|pymupdf]
param(
    [switch]$SkipTrain,
    [ValidateSet("mineru", "pymupdf")][string]$Engine,
    [string]$Version,
    [string]$Python = "python"
)

Set-Location $PSScriptRoot

function Step($text) {
    Write-Host ""
    Write-Host "=============================================================="
    Write-Host ">> $text"
    Write-Host "=============================================================="
}

function Invoke-Step($label, $scriptArgs) {
    Step $label
    & $Python @scriptArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Error "FAILED at: $label (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}

# Inference failures must not hide the comparison: an unparseable output is itself
# a result the verdict should record, so these steps warn and continue.
function Invoke-SoftStep($label, $scriptArgs) {
    Step $label
    & $Python @scriptArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$label exited $LASTEXITCODE - continuing so compare.py can record it"
    }
}

$ocrArgs = @("src/run_ocr.py")
if ($Engine) { $ocrArgs += @("--engine", $Engine) }
# Which fine-tuned version this run produces. Defaults to model.version in config.yaml.
$verArgs = @()
if ($Version) { $verArgs = @("--version", $Version) }
$modelArg = $Version
if (-not $modelArg) {
    $modelArg = & $Python -c "import sys;sys.path.insert(0,'src');from common import load_config,model_version;print(model_version(load_config()))"
}
Write-Host "fine-tuned version for this run: $modelArg"

# build_dataset.py OCRs any document that needs it, so OCR is not a separate step.
Invoke-Step "1/7 build training corpus (OCRs as needed)" @("src/build_dataset.py")
# Audit the ground truth before training on it. Runs after OCR so it can check
# that every value in the gold actually appears on its page.
Invoke-SoftStep "2/7 audit the golden JSON" @("src/verify_gold.py", "--ocr")
# The base extraction is taken BEFORE training, so the "before" measurement exists
# on disk no matter what happens during the training run.
Invoke-SoftStep "3/7 extract with BASE model" @("src/infer.py", "--model", "base")
if (-not $SkipTrain) {
    Invoke-Step "4/7 QLoRA fine-tune" (@("src/train.py") + $verArgs)
    Invoke-Step "5/7 merge adapter" (@("src/merge.py", "--force") + $verArgs)
} else {
    Step "4-5/7 training and merge skipped (-SkipTrain)"
}
Invoke-SoftStep "6/7 extract with the fine-tuned model" @("src/infer.py", "--model", $modelArg)

Step "7/7 compare base vs the fine-tuned model"
& $Python src/compare.py @verArgs
$verdict = $LASTEXITCODE

Write-Host ""
if ($verdict -eq 0) {
    Write-Host "POC RESULT: PASS - $modelArg is closer to the gold JSON than the base model."
} else {
    Write-Host "POC RESULT: INVESTIGATE - see results/trained model results/<version>/<doc>/comparison.json and the reasons above."
}
exit $verdict
