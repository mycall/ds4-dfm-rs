# Manage the native N1X CLI and resident GGUF endpoint.
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet('status', 'build', 'test', 'run', 'start', 'stop', 'help')]
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
$Backend = $Backend.ToLowerInvariant()
$exe = Join-Path $PSScriptRoot 'target/n1x/ds4.exe'
$server = Join-Path $PSScriptRoot 'target/n1x/ds4-gguf-server.exe'
$url = 'http://127.0.0.1:8081'
$log = Join-Path $PSScriptRoot 'target/n1x/http.stderr.log'
function Get-ResidentServer {
    Get-Process -Name ds4-gguf-server -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $server }
}
switch ($Action) {
    'help' {
        @'
.\ds4-n1x.ps1 status                       Show executable, first shard, and active CLI processes
.\ds4-n1x.ps1 start                        Serve the model on localhost:8081 (default context 32768)
.\ds4-n1x.ps1 stop                         Release the resident model and GPU cache
.\ds4-n1x.ps1 build                        Build the native hosts and run model-free smoke checks
.\ds4-n1x.ps1 test                         Test the resident HTTP server, or the CLI when stopped
.\ds4-n1x.ps1 run "What is two plus two?"  Generate one answer and exit

Start, run and test accept -Backend cuda|cpu and -Model <first-shard.gguf>.
Run also accepts -MaxTokens (default 64) and -Context (default 512).
'@
        return
    }
    'status' {
        $running = Get-Process -Name ds4 -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe }
        $resident = Get-ResidentServer
        $ready = $false
        if ($resident) {
            try { $ready = (Invoke-RestMethod "$url/health" -TimeoutSec 2).status -eq 'ok' } catch {}
        }
        [pscustomobject]@{
            ServerPid = @($resident | Select-Object -ExpandProperty Id) -join ', '
            Endpoint = "$url/v1"
            Ready = $ready
            ServerLog = $log
            Executable = $exe
            Built = Test-Path -LiteralPath $exe -PathType Leaf
            FirstShard = $Model
            FirstShardPresent = Test-Path -LiteralPath $Model -PathType Leaf
            RunningPids = @($running | Select-Object -ExpandProperty Id) -join ', '
        }
        return
    }
    'stop' {
        $resident = Get-ResidentServer
        if ($resident) {
            $resident | Stop-Process -ErrorAction Stop
            foreach ($p in $resident) { $p.WaitForExit() }
            'Stopped resident GGUF server.'
        } else { 'Resident GGUF server is stopped.' }
        return
    }
    'start' {
        if (Get-ResidentServer) { throw 'The server is already running. Use status or stop.' }
        if (Get-Process -Name ds4 -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe }) {
            throw 'Wait for the active one-shot generation to finish before starting the resident server.'
        }
        if (!(Test-Path -LiteralPath $server -PathType Leaf)) { throw 'Run .\ds4-n1x.ps1 build first.' }
        $shard = (Get-Item -LiteralPath $Model -ErrorAction Stop).FullName
        $network = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties()
        if ($network.GetActiveTcpListeners().Port -contains 8081) {
            throw 'Port 8081 is in use.'
        }
        if (!$PSBoundParameters.ContainsKey('Context')) { $Context = 32768 }
        $launch = @{
            FilePath = $server
            ArgumentList = @(('"{0}"' -f $shard), $Backend, $Context, 8081)
            WindowStyle = 'Hidden'
            RedirectStandardError = $log
            RedirectStandardOutput = "$log.stdout"
            PassThru = $true
        }
        $process = Start-Process @launch
        Write-Host "Loading the model (PID $($process.Id)). Log: $log"
        $deadline = (Get-Date).AddMinutes(5)
        while (!$process.HasExited -and (Get-Date) -lt $deadline) {
            try {
                if ((Invoke-RestMethod "$url/health" -TimeoutSec 2).status -eq 'ok') {
                    Write-Host "Ready: $url/v1"
                    return
                }
            } catch {}
            Start-Sleep -Milliseconds 500
        }
        if ($process.HasExited) { throw "Server exited with $($process.ExitCode). See $log" }
        throw "Server is still loading. Use status or inspect $log."
    }
    'build' {
        if (Get-ResidentServer) { throw 'Stop the resident server before rebuilding its DLLs.' }
        & "$PSScriptRoot/native/windows/build.ps1" -Smoke
        return
    }
}

if ($Action -eq 'run' -and [string]::IsNullOrWhiteSpace($Prompt)) {
    throw 'Supply a prompt, for example: .\ds4-n1x.ps1 run "What is two plus two?"'
}
if (Get-ResidentServer) {
    if ($PSBoundParameters.ContainsKey('Model') -or $PSBoundParameters.ContainsKey('Backend') -or
        $PSBoundParameters.ContainsKey('Context')) { throw 'Choose the model, backend and context when starting the server.' }
    if ($Action -eq 'test') {
        & python "$PSScriptRoot/native/windows/smoke-http.py"
        exit $LASTEXITCODE
    }
    $request = @{
        model = 'DeepSeek-V4.1-Flash-1017-unsloth-UD-IQ2_XXS'
        messages = @(@{role = 'user'; content = $Prompt})
        max_tokens = $MaxTokens
        temperature = 0
    } | ConvertTo-Json -Depth 4 -Compress
    $answer = Invoke-RestMethod "$url/v1/chat/completions" -Method Post -Body $request -ContentType 'application/json; charset=utf-8' -TimeoutSec 3600
    $answer.choices[0].message.content
    return
}
if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw 'Run .\ds4-n1x.ps1 build first.' }
if (!(Test-Path -LiteralPath $Model -PathType Leaf)) { throw "First shard missing: $Model" }
if ($Action -eq 'test') {
    & python "$PSScriptRoot/native/windows/smoke-gguf.py" $exe $Model --backend $Backend
} else {
    & $exe -m $Model --backend $Backend --ctx $Context --threads 12 --nothink --temp 0 -n $MaxTokens -p $Prompt
}
exit $LASTEXITCODE
