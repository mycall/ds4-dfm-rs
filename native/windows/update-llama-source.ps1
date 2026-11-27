# Prepare the pinned source tree used by the N1X GGUF build.
param([Parameter(Mandatory)][string]$Src)
$ErrorActionPreference = 'Stop'
$Src = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Src)

function Invoke-Git([string[]]$Arguments) {
    & git @Arguments
    if ($LASTEXITCODE) { throw "git $($Arguments[0]) exited with $LASTEXITCODE" }
}

$pin = '244640b7f26e2a7301b27a24edc4f7feaf97d52f'
if (!(Test-Path "$src/.git")) {
    Invoke-Git @('init', $src)
    Invoke-Git @('-C', $src, 'fetch', '--depth=1', 'https://github.com/ggml-org/llama.cpp.git', $pin)
    Invoke-Git @('-C', $src, '-c', 'core.longpaths=true', 'checkout', '--detach', 'FETCH_HEAD')
}
if ((Invoke-Git @('-C', $src, 'rev-parse', 'HEAD')) -ne $pin) { throw "Expected llama.cpp $pin in $src" }
# Older N1X builds copied headers into src without tracking them in Git.
# Remember the installed blobs separately from the developer's index.
$receipt = Invoke-Git @('-C', $Src, 'rev-parse', '--path-format=absolute', '--git-path', 'ds4-n1x-headers')
$installed = @{}
if (Test-Path -LiteralPath $receipt) {
    $installed = ConvertFrom-StringData ([IO.File]::ReadAllText($receipt))
}
$headers = @('ds41_cache_policy.hpp', 'ds41_expert_read.hpp', 'ds41_expert_upload.hpp')
$copies = @{}
$records = foreach ($name in $headers) {
    $source = Join-Path $PSScriptRoot $name
    $dest = Join-Path $Src "src/$name"
    $hashArgs = @('-C', $PSScriptRoot, '-c', 'core.autocrlf=input', 'hash-object', "--path=$name")
    $wanted = Invoke-Git ($hashArgs + $source)
    if (Test-Path -LiteralPath $dest) {
        $item = Get-Item -Force -LiteralPath $dest
        if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Unrecognized changes in $dest"
        }
        $present = Invoke-Git ($hashArgs + $dest)
        if ($present -ne $wanted) {
            if ($installed.ContainsKey($name)) {
                $known = $present -eq $installed[$name]
            } else {
                # Migrate a copy made before receipts existed if it matches
                # a published version of this exact header in the N1X repo.
                $versions = Invoke-Git @('-C', $PSScriptRoot, 'rev-list', '--objects', '--no-object-names', 'HEAD', '--', $name)
                $known = $versions -contains $present
            }
            if (!$known) { throw "Unrecognized changes in $dest" }
            $copies[$name] = $present
        }
    } else {
        $copies[$name] = $null
    }
    "$name=$wanted"
}
$patches = @('ds41-demand-map.patch', 'ds41-http.patch', 'ds41-wide-prefetch.patch',
             'ds41-seed-experts.patch', 'ds41-expert-io.patch', 'ds41-cache-policies.patch',
             'ds41-layer-loop.patch', 'ds41-loop-expert-cache.patch', 'ds41-loop-depth.patch') |
    ForEach-Object { Join-Path $PSScriptRoot $_ }
# A later patch can change lines added by an earlier one. Compare the checkout
# to the known prefixes of the patch stack without touching the developer's index.
$paths = @($patches | ForEach-Object {
    Invoke-Git @('-C', $src, 'apply', '--numstat', $_) | ForEach-Object { ($_ -split "`t", 3)[2] }
} | Sort-Object -Unique)
$previousIndex = $env:GIT_INDEX_FILE
$tempIndex = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
$normalized = "$tempIndex.patch"
$matched = $false
try {
    $env:GIT_INDEX_FILE = $tempIndex
    Invoke-Git @('-C', $src, 'read-tree', $pin)
    for ($i = 0; $i -le $patches.Count; $i++) {
        if (!$matched) {
            git -C $src diff --quiet -- $paths
            if ($LASTEXITCODE -gt 1) { throw 'Could not compare the llama.cpp checkout' }
            $matched = $LASTEXITCODE -eq 0
        }
        if ($i -eq $patches.Count) { break }
        $patchText = [IO.File]::ReadAllText($patches[$i]).Replace("`r`n", "`n")
        [IO.File]::WriteAllText($normalized, $patchText)
        if ($matched) {
            Invoke-Git @('-C', $src, 'apply', $normalized)
        } else {
            Invoke-Git @('-C', $src, 'apply', '--cached', $normalized)
        }
    }
    if (!$matched) { throw "Unrecognized changes in the patched llama.cpp files in $src" }
} finally {
    $env:GIT_INDEX_FILE = $previousIndex
    [IO.File]::Delete($tempIndex)
    [IO.File]::Delete($normalized)
}
foreach ($name in $copies.Keys) {
    $source = Join-Path $PSScriptRoot $name
    $dest = Join-Path $Src "src/$name"
    if ($null -eq $copies[$name]) {
        [IO.File]::Copy($source, $dest)
        continue
    }
    $tempHeader = Join-Path (Split-Path -Parent $dest) ([IO.Path]::GetRandomFileName())
    try {
        [IO.File]::Copy($source, $tempHeader)
        $present = Invoke-Git @('-C', $PSScriptRoot, '-c', 'core.autocrlf=input', 'hash-object', "--path=$name", $dest)
        if ($present -ne $copies[$name]) { throw "Unrecognized changes in $dest" }
        [IO.File]::Replace($tempHeader, $dest, [NullString]::Value)
    } finally {
        [IO.File]::Delete($tempHeader)
    }
}
$record = ($records -join "`n") + "`n"
if (!(Test-Path -LiteralPath $receipt) -or [IO.File]::ReadAllText($receipt) -cne $record) {
    $tempReceipt = Join-Path (Split-Path -Parent $receipt) ([IO.Path]::GetRandomFileName())
    try {
        [IO.File]::WriteAllText($tempReceipt, $record)
        if (Test-Path -LiteralPath $receipt) {
            [IO.File]::Replace($tempReceipt, $receipt, [NullString]::Value)
        } else {
            [IO.File]::Move($tempReceipt, $receipt)
        }
    } finally {
        [IO.File]::Delete($tempReceipt)
    }
}
