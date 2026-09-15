param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Arguments)
$ErrorActionPreference = "Stop"
$runner = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path "run.py"
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $runner translate-all @Arguments
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $runner translate-all @Arguments
} else {
    throw "Python 3 was not found."
}
exit $LASTEXITCODE
