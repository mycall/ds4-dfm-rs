# Compare fresh four-slot N1X CUDA servers on the same cold, multi-batch prompt.
param([Parameter(Mandatory)][string]$Model)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot/../..").Path
$launcher = Join-Path $root 'ds4-n1x.ps1'
$Model = (Get-Item -LiteralPath $Model).FullName
$excerpt = [IO.File]::ReadAllText((Join-Path $root 'AGENTS.md')).Replace("`r`n", "`n")
$prompt = "Read these repository instructions.`n" + $excerpt.Substring(0, [Math]::Min(14500, $excerpt.Length)) +
          "`nReply with just the digit 4."
$payload = @{
    model = 'DeepSeek-V4.1-Flash-1017-unsloth-UD-IQ2_XXS'
    messages = @(@{ role = 'user'; content = $prompt })
    temperature = 0
    seed = 42
    max_tokens = 1
} | ConvertTo-Json -Depth 4 -Compress

function Measure-Prefill($label) {
    & $launcher start -Model $Model | Out-Host
    try {
        $serverId = (& $launcher status).ServerPid
        $filter = "ProcessId=$serverId"
        $before = (Get-CimInstance Win32_Process -Filter $filter).ReadTransferCount
        $reply = Invoke-RestMethod 'http://127.0.0.1:8081/v1/chat/completions' -Method Post -Body $payload `
            -ContentType 'application/json; charset=utf-8' -TimeoutSec 900
        $after = (Get-CimInstance Win32_Process -Filter $filter).ReadTransferCount
        $n = $reply.timings.prompt_n
        if ($reply.timings.cache_n -ne 0 -or $n -le 2048 -or $after -le $before) {
            throw 'Expected uncached prefill over multiple 2,048-row batches and measurable server reads.'
        }
        [pscustomobject]@{ Policy = $label; PromptTokens = $n; ReadBytes = $after - $before
            PrefillMs = $reply.timings.prompt_ms; Answer = $reply.choices[0].message.content }
    } finally { & $launcher stop | Out-Host }
}

$seed = $env:DS4_N1X_SEED_EXPERTS
$direct = $env:DS4_N1X_DIRECT_EXPERT_IO
try {
    $env:DS4_N1X_DIRECT_EXPERT_IO = '1'
    $env:DS4_N1X_SEED_EXPERTS = '0'
    $off = Measure-Prefill 'disabled'
    Remove-Item Env:DS4_N1X_SEED_EXPERTS
    $default = Measure-Prefill 'default'
    $off, $default | Format-Table | Out-Host
    if ($default.PromptTokens -ne $off.PromptTokens -or $default.Answer -ne $off.Answer -or
        $default.Answer.Trim() -ne '4' -or $default.ReadBytes -ge $off.ReadBytes) {
        throw 'Expected the same answer and fewer server read bytes for the same prompt.'
    }
} finally {
    $env:DS4_N1X_SEED_EXPERTS = $seed
    $env:DS4_N1X_DIRECT_EXPERT_IO = $direct
}
