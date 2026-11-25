# Manage the native N1X one-shot CLI. Run .\ds4-n1x.ps1 help for examples.
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet('status', 'build', 'test', 'run', 'help')]
    [string]$Action = 'status',
    [Parameter(Position = 1)]
    [string]$Prompt,
    [string]$Model = "$HOME/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/unsloth/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf",
    [ValidateSet('cuda', 'cpu')]
    [string]$Backend = 'cuda',
    [ValidateRange(1, 2147483647)]
    [int]$MaxTokens = 64,
    [ValidateRange(1, 2147483647)]
    [int]$Context = 512
)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'target/n1x/ds4.exe'
switch ($Action) {
    'help' {
        @'
.\ds4-n1x.ps1 status                       Show executable, first shard, and active CLI processes
.\ds4-n1x.ps1 build                        Build the native hosts and run model-free smoke checks
.\ds4-n1x.ps1 test                         Run the guarded arithmetic test on the actual model
.\ds4-n1x.ps1 run "What is two plus two?"  Generate one answer and exit

Run and test accept -Backend cuda|cpu and -Model <first-shard.gguf>.
Run also accepts -MaxTokens (default 64) and -Context (default 512).
'@
        return
    }
    'status' {
        $running = Get-Process -Name ds4 -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe }
        [pscustomobject]@{
            Executable = $exe
            Built = Test-Path -LiteralPath $exe -PathType Leaf
            FirstShard = $Model
            FirstShardPresent = Test-Path -LiteralPath $Model -PathType Leaf
            RunningPids = @($running | Select-Object -ExpandProperty Id) -join ', '
        }
        return
    }
    'build' {
        & "$PSScriptRoot/native/windows/build.ps1" -Smoke
        return
    }
}

if ($Action -eq 'run' -and [string]::IsNullOrWhiteSpace($Prompt)) {
    throw 'Supply a prompt, for example: .\ds4-n1x.ps1 run "What is two plus two?"'
}
if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw 'Run .\ds4-n1x.ps1 build first.' }
if (!(Test-Path -LiteralPath $Model -PathType Leaf)) { throw "First shard missing: $Model" }
if ($Action -eq 'test') {
    & python "$PSScriptRoot/native/windows/smoke-gguf.py" $exe $Model --backend $Backend
} else {
    & $exe -m $Model --backend $Backend --ctx $Context --threads 12 --nothink --temp 0 -n $MaxTokens -p $Prompt
}
exit $LASTEXITCODE
