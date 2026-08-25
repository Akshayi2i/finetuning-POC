# End-to-end POC run for Windows hosts. Mirrors run_all.sh.
# Usage: .\run_all.ps1 [-SkipTrain] [-Engine mineru|pymupdf]
param(
    [switch]$SkipTrain,
    [ValidateSet("mineru", "pymupdf")][string]$Engine,
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

Invoke-Step "1/8 OCR every document" $ocrArgs
# Audit the ground truth before training on it. Runs after OCR so it can check
# that every value in the gold actually appears on its page.
Invoke-SoftStep "2/8 audit the golden JSON" @("src/verify_gold.py", "--ocr")
Invoke-Step "3/8 build training corpus" @("src/build_dataset.py")
# The base extraction is taken BEFORE training, so the "before" measurement exists
# on disk no matter what happens during the training run.
Invoke-SoftStep "4/8 extract with BASE model" @("src/infer.py", "--model", "base")
if (-not $SkipTrain) {
    Invoke-Step "5/8 QLoRA fine-tune" @("src/train.py")
    Invoke-Step "6/8 merge adapter" @("src/merge.py", "--force")
} else {
    Step "5-6/8 training and merge skipped (-SkipTrain)"
}
Invoke-SoftStep "7/8 extract with V1 (fine-tuned)" @("src/infer.py", "--model", "v1")

Step "8/8 compare base vs v1"
& $Python src/compare.py
$verdict = $LASTEXITCODE

Write-Host ""
if ($verdict -eq 0) {
    Write-Host "POC RESULT: PASS - v1 is closer to the gold JSON than the base model."
} else {
    Write-Host "POC RESULT: INVESTIGATE - see outputs/results/comparison.json and the reasons above."
}
exit $verdict
