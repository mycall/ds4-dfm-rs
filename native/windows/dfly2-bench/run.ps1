param(
    [Parameter(Mandatory)][ValidateSet('dfly2','dspark','plain','replay')][string]$Mode,
    [Parameter(Mandatory)][string]$Target,
    [string]$Draft, [string]$ReplayLog,
    [ValidateRange(2,32767)][int]$Tokens = 64,
    [ValidateRange(1,100)][int]$Repeats = 3,
    [ValidateRange(1,98304)][int]$CacheMiB = 69632,
    [ValidateRange(1,5)][int]$DraftMax = 5,
    [ValidateNotNullOrEmpty()][string]$Prompt,
    [ValidateRange(32,32768)][int]$Context = 32768,
    [switch]$ShortOnly
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot/../../..").Path
$out = "$root/target/n1x-dfly2-bench"
$exe = "$out/probe.exe"
if (!(Test-Path -LiteralPath $exe)) { throw 'Run the adjacent build.ps1 first' }
$targetPath = (Resolve-Path -LiteralPath $Target).Path
$draftPath = '-'
if ($Mode -in @('dfly2','dspark')) {
    $draftPath = (Resolve-Path -LiteralPath $Draft).Path
    if ((Test-Path -LiteralPath $draftPath -PathType Container) -ne ($Mode -eq 'dfly2')) {
        throw 'DFly2 takes a safetensors directory and DSpark takes a GGUF file'
    }
}

$prefix = "$out/$Mode-$Tokens-$(Get-Date -Format yyyyMMdd-HHmmss-fff)"
$inputs = "$prefix.prompts.json"
if ($Mode -eq 'replay') {
    $cases = @(Get-Content -LiteralPath $ReplayLog | ForEach-Object { ConvertFrom-Json -InputObject $_ })
    if (!$cases.Count) { throw 'The replay log contains no requests' }
    foreach ($case in $cases) {
        $budget = if ($null -ne $case.output_budget) { $case.output_budget } else { $case.tokens.Count }
        if ($budget -ne $Tokens) { throw 'Use the original output budget with -Tokens' }
    }
    [IO.File]::WriteAllText($inputs, (ConvertTo-Json -InputObject $cases -Compress -Depth 10))
} elseif ($PSBoundParameters.ContainsKey('Prompt')) {
    $cases = @(foreach ($i in 1..$Repeats) { @{ user_prompt = $Prompt } })
    [IO.File]::WriteAllText($inputs, (ConvertTo-Json -InputObject $cases -Compress))
} else {
    $lf = [char]10
    $prompts = @(
        "# Write a Python function that merges overlapping intervals. Include type annotations and two tests.${lf}from typing import Iterable${lf}${lf}def merge_intervals(",
        "A new engineer asks why a database needs a write-ahead log. Explain with a concrete example of a power failure halfway through a transaction.${lf}${lf}A write-ahead log"
    )
    if (!$ShortOnly) {
        $code = [IO.File]::ReadAllText("$root/native/windows/ds41_http.cpp").Replace("`r`n", "`n")
        $prompts += $code + $lf + '/* Explain how the server chooses its CUDA expert cache size. Give a worked example for a 96 GiB GPU, and explain the host memory watchdog.' + $lf
    }
    [IO.File]::WriteAllText($inputs, (ConvertTo-Json -InputObject @($prompts * $Repeats) -Compress))
}

$settings = @{
    PATH = "$root/target/n1x;$env:PATH"
    DS4_N1X_DIRECT_EXPERT_IO = '1'
    DS4_N1X_PREFETCH_EXPERTS = '0'
    DS4_N1X_SEED_EXPERTS = 'learned'
}
$saved = @{}
try {
    foreach ($name in $settings.Keys) {
        $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $settings[$name], 'Process')
    }
    & $exe $targetPath $draftPath $inputs $Tokens $CacheMiB $DraftMax $Context >"$prefix.jsonl" 2>"$prefix.log"
    if ($LASTEXITCODE) { throw "Benchmark exited with $LASTEXITCODE. See $prefix.log" }
} finally {
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
Write-Output "$prefix.jsonl"
Get-Content -LiteralPath "$prefix.jsonl" | ForEach-Object { ConvertFrom-Json -InputObject $_ } |
    Select-Object logical_layers, prompt_tokens, answer, tps, prefill_s, decode_s, accepted, proposed
