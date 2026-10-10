# Regenerate the MATLAB R2026a goldens G1-G8 (DESIGN.md section 10.6; WP4).
#
# Usage (from any directory):
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\matlab\run_goldens.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\matlab\run_goldens.ps1 -Only G4,G8
#
# MATLAB R2026b is first on PATH on the author's laptop, so this script calls the R2026a
# executable explicitly. Override with -Matlab or $env:SREMU_MATLAB_R2026A. Outputs go to
# tests\golden\matlab_R2026a (CSV + meta.json per golden). MATLAB is not a CI dependency.
param(
    [string[]]$Only = @(),
    [string]$Matlab = $(if ($env:SREMU_MATLAB_R2026A) { $env:SREMU_MATLAB_R2026A } else { "C:\Program Files\MATLAB\R2026a\bin\matlab.exe" }),
    [string]$OutDir = ""
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $here "..\..")
if (-not $OutDir) { $OutDir = Join-Path $repo "tests\golden\matlab_R2026a" }
if (-not (Test-Path $Matlab)) {
    Write-Error "MATLAB R2026a not found at '$Matlab' (set -Matlab or SREMU_MATLAB_R2026A)"
    exit 2
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$hereM = $here -replace "'", "''"
$outM = $OutDir -replace "'", "''"
# -File passes "-Only G4,G8" as one string; split on commas either way
$Only = @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($Only.Count -gt 0) {
    $ids = ($Only | ForEach-Object { "'" + $_.Trim() + "'" }) -join ","
    $call = "golden_run_all('$outM', {$ids})"
} else {
    $call = "golden_run_all('$outM')"
}
$cmd = "addpath('$hereM'); $call"
Write-Host "[run_goldens] $Matlab -batch `"$cmd`""
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $Matlab -batch $cmd
$rc = $LASTEXITCODE
$sw.Stop()
Write-Host ("[run_goldens] MATLAB exit code {0}, wall {1:N1} s" -f $rc, $sw.Elapsed.TotalSeconds)
exit $rc
