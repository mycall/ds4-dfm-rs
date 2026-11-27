# Build against the existing N1X CUDA engine, including its Rust expert cache policy.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot/../../..").Path
$source = "$root/target/n1x/llama-src"
$build = "$root/target/n1x/llama-cuda"
$out = (New-Item -ItemType Directory -Force "$root/target/n1x-dfly2-bench").FullName
$stage = (New-Item -ItemType Directory -Force "$out/source").FullName

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE) { throw "$Program exited with $LASTEXITCODE" }
}

$vswhere = Join-Path ([Environment]::GetFolderPath('ProgramFilesX86')) 'Microsoft Visual Studio/Installer/vswhere.exe'
$install = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
$setup = Join-Path $install 'VC/Auxiliary/Build/vcvarsarm64.bat'
$vars = & cmd.exe /d /c ('call "{0}" >nul && set' -f $setup)
if ($LASTEXITCODE) { throw 'Cannot load the native ARM64 build environment' }
foreach ($line in $vars) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$cc = Join-Path $env:VCINSTALLDIR 'Tools/Llvm/ARM64/bin/clang-cl.exe'
if (!(Test-Path "$build/src/llama.lib")) { throw 'Build the N1X CUDA GGUF engine first' }

# Only the private benchmark DLL exports target intermediates. Reuse the remaining build objects.
Invoke-Checked git @('-C', $stage, 'init', '-q')
foreach ($file in @('src/llama-context.cpp', 'src/llama-context.h', 'src/models/deepseek41.cpp')) {
    $dest = Join-Path $stage $file
    $null = New-Item -ItemType Directory -Force (Split-Path $dest)
    Copy-Item -LiteralPath "$source/$file" -Destination $dest -Force
}
Invoke-Checked git @('-C', $stage, 'apply', '--ignore-space-change', "$PSScriptRoot/dfly-outputs.patch")
$unity = @(Get-ChildItem "$build/src" -Filter 'unity_*_cxx.cxx' -Recurse |
    Where-Object { Select-String -Path $_.FullName -Pattern '/models/deepseek41\.cpp"' -Quiet })
if ($unity.Count -ne 1) { throw 'Expected one CMake unity source containing deepseek41.cpp' }
$unityName = $unity[0].Name
$content = [IO.File]::ReadAllText($unity[0].FullName)
$content = $content -replace '(?m)^#include "[^"]*/models/deepseek41\.cpp"', '#include "models/deepseek41.cpp"'
[IO.File]::WriteAllText("$stage/src/$unityName", $content)
$flags = @('/nologo', '/O2', '/MD', '/EHsc', '/clang:-mcpu=native', '/DNDEBUG', '/D_CRT_SECURE_NO_WARNINGS',
           "/I$PSScriptRoot", "/I$stage/src", "/I$source/src", "/I$source/include", "/I$source/ggml/include")
$dllFlags = @('/c', '/TP', '/std:c++17', '/bigobj', '/utf-8', '/DGGML_BACKEND_SHARED', '/DGGML_SHARED',
    '/DGGML_USE_CPU', '/DGGML_USE_CUDA', '/DLLAMA_BUILD', '/DLLAMA_SHARED', "/I$source/src/models", "/I$build/src")
$replace = @('llama-context.cpp', $unityName)
foreach ($file in $replace) {
    Invoke-Checked $cc ($flags + $dllFlags + @("$stage/src/$file", "/Fo$out/$file.obj"))
}
$objects = @(Get-ChildItem -LiteralPath "$build/src/CMakeFiles/llama.dir" -Recurse -Filter '*.obj' |
    ForEach-Object { if ($_.BaseName -in $replace) { "$out/$($_.Name)" } else { $_.FullName } })
$ggml = "$build/ggml/src"
$link = @('/NOLOGO', '/DLL', '/OPT:REF', "/OUT:$out/llama.dll", "/IMPLIB:$out/llama.lib") + $objects +
    @("$ggml/ggml.lib", "$build/bin/ds4_n1x_cache.dll.lib", "$ggml/ggml-cpu.lib", "$ggml/ggml-cuda/ggml-cuda.lib",
      "$ggml/ggml-base.lib", 'kernel32.lib', 'user32.lib', 'gdi32.lib', 'advapi32.lib', 'shell32.lib',
      'ole32.lib', 'oleaut32.lib', 'uuid.lib', 'comdlg32.lib', 'winspool.lib')
[IO.File]::WriteAllLines("$out/link.rsp", @($link | ForEach-Object { '"' + $_ + '"' }))
Invoke-Checked link.exe @("@$out/link.rsp")
Invoke-Checked $cc ($flags + @('/std:c++20', "/I$root/native/windows", "/I$source/common", "/I$source/vendor",
    "$PSScriptRoot/probe.cpp", "/Fo$out/probe.obj", '/link', "/OUT:$out/probe.exe", "$build/common/llama-common.lib",
    "$out/llama.lib", "$ggml/ggml-base.lib", "$ggml/ggml.lib", 'psapi.lib'))
