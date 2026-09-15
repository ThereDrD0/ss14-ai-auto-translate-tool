param(
    [switch] $DryRun,
    [int] $BatchSize = 100,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($BatchSize -lt 1) {
    throw "BatchSize must be 1 or greater."
}

$toolRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $toolRoot "run.py"
$arguments = @(
    "translate-all",
    "--source-culture", "ru-RU",
    "--target-culture", "en-US",
    "--batch-size", $BatchSize
)

if ($DryRun) {
    $arguments += "--dry-run"
}

$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    & $python.Source $runner @arguments @ExtraArguments
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $runner @arguments @ExtraArguments
} else {
    throw "Python 3 was not found."
}

exit $LASTEXITCODE
