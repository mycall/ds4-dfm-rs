# Called by build.ps1 inside its native ARM64 VS environment.
param([Parameter(Mandatory)][string]$Out, [switch]$CpuOnly, [switch]$Smoke)
$ErrorActionPreference = 'Stop'
$pin = '244640b7f26e2a7301b27a24edc4f7feaf97d52f'
$src = Join-Path $Out 'llama-src'
if (!(Test-Path "$src/.git")) {
    Invoke-Checked git @('init', $src)
    Invoke-Checked git @('-C', $src, 'fetch', '--depth=1', 'https://github.com/ggml-org/llama.cpp.git', $pin)
    Invoke-Checked git @('-C', $src, '-c', 'core.longpaths=true', 'checkout', '--detach', 'FETCH_HEAD')
}
if ((& git -C $src rev-parse HEAD) -ne $pin) { throw "Expected llama.cpp $pin in $src" }
$patches = @('ds41-demand-map.patch', 'ds41-http.patch', 'ds41-wide-prefetch.patch',
             'ds41-seed-experts.patch', 'ds41-expert-io.patch', 'ds41-cache-policies.patch') |
    ForEach-Object { Join-Path $PSScriptRoot $_ }
# A later patch can change lines added by an earlier one. Compare the checkout
# to the known prefixes of the patch stack without touching the developer's index.
$paths = @($patches | ForEach-Object {
    git -C $src apply --numstat $_ | ForEach-Object { ($_ -split "`t", 3)[2] }
} | Sort-Object -Unique)
$previousIndex = $env:GIT_INDEX_FILE
$tempIndex = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
$normalized = "$tempIndex.patch"
$matched = $false
try {
    $env:GIT_INDEX_FILE = $tempIndex
    Invoke-Checked git @('-C', $src, 'read-tree', $pin)
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
            Invoke-Checked git @('-C', $src, 'apply', $normalized)
        } else {
            Invoke-Checked git @('-C', $src, 'apply', '--cached', $normalized)
        }
    }
    if (!$matched) { throw "Unrecognized changes in the patched llama.cpp files in $src" }
} finally {
    $env:GIT_INDEX_FILE = $previousIndex
    [IO.File]::Delete($tempIndex)
    [IO.File]::Delete($normalized)
}
Copy-Item -LiteralPath "$PSScriptRoot/ds41_cache_policy.hpp" -Destination "$src/src/ds41_cache_policy.hpp" -Force
Copy-Item -LiteralPath "$PSScriptRoot/ds41_expert_upload.hpp" -Destination "$src/src/ds41_expert_upload.hpp" -Force
Copy-Item -LiteralPath "$PSScriptRoot/ds41_expert_read.hpp" -Destination "$src/src/ds41_expert_read.hpp" -Force
$cuda = if ($CpuOnly) { 'OFF' } else { 'ON' }
$build = Join-Path $Out $(if ($CpuOnly) { 'llama-build' } else { 'llama-cuda' })
$llvm = Join-Path $env:VCINSTALLDIR 'Tools/Llvm/ARM64/bin'
$cmakeTools = Join-Path $env:VSINSTALLDIR 'Common7/IDE/CommonExtensions/Microsoft/CMake'
$env:Path = "$llvm;$cmakeTools/CMake/bin;$cmakeTools/Ninja;$env:Path"
$runtimeBin = (New-Item -ItemType Directory -Force (Join-Path $build 'bin')).FullName
$policy = Join-Path $runtimeBin 'ds4_n1x_cache.dll'
Invoke-Checked rustc @('--edition=2021', '--target', 'aarch64-pc-windows-msvc', '--crate-type=cdylib',
    '-C', 'opt-level=3', '-C', 'panic=abort', "$PSScriptRoot/cache_policy.rs", '-o', $policy)
$options = @('-G', 'Ninja', "-DDS4_N1X_POLICY_LIB=$($policy.Replace('\','/')).lib", '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=clang-cl',
    '-DCMAKE_CXX_COMPILER=clang-cl', '-DCMAKE_C_FLAGS=/clang:-mcpu=native',
    '-DCMAKE_CXX_FLAGS=/EHsc /clang:-mcpu=native', '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF',
    '-DGGML_CPU_KLEIDIAI=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
    '-DLLAMA_BUILD_COMMON=ON', '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_SERVER=ON',
    '-DLLAMA_BUILD_APP=OFF', '-DLLAMA_BUILD_UI=OFF', '-DLLAMA_OPENSSL=OFF', '-DLLAMA_USE_PREBUILT_UI=OFF', "-DGGML_CUDA=$cuda")
if (!$CpuOnly) { $options += @('-DCMAKE_CUDA_ARCHITECTURES=121a', "-DCMAKE_CUDA_HOST_COMPILER=$((Get-Command cl.exe).Source)") }
Invoke-Checked cmake (@('-S', $src, '-B', $build) + $options)
Invoke-Checked cmake @('--build', $build, '--target', 'llama-server-impl', '--parallel', '10')
# Separate DLL keeps the engine's vendored ggml stubs isolated from llama.cpp.
Invoke-Checked clang-cl @('/nologo', '/LD', '/O2', '/MD', '/clang:-mcpu=native', '/std:c++17', '/EHs',
    "-I$PSScriptRoot/../..", "-I$src/include", "-I$src/ggml/include", "$PSScriptRoot/ds41_gguf.cpp", "/Fo$Out/ds41_gguf.obj",
    '/link', "/OUT:$Out/ds41_gguf.dll", "/IMPLIB:$Out/ds41_gguf.lib",
    "$build/src/llama.lib", "$build/ggml/src/ggml.lib", "$build/ggml/src/ggml-base.lib")
Invoke-Checked clang-cl @('/nologo', '/O2', '/MD', '/clang:-mcpu=native', '/std:c++17', '/EHsc',
    "-I$src/include", "-I$src/ggml/include", "$PSScriptRoot/ds41_http.cpp", "/Fo$Out/ds41_http.obj",
    '/link', "/OUT:$Out/ds4-gguf-server.exe", "$build/tools/server/llama-server-impl.lib",
    "$build/src/llama.lib", "$build/ggml/src/ggml.lib", "$build/ggml/src/ggml-base.lib")
Copy-Item "$build/bin/*.dll" -Destination $Out -Force
if ($Smoke) {
    Invoke-Checked rustc @('--edition=2021', '--target', 'aarch64-pc-windows-msvc', '--test',
        "$PSScriptRoot/cache_policy.rs", '-o', "$Out/test_cache_policy.exe")
    Invoke-Checked "$Out/test_cache_policy.exe" @()
    $objs = "$build/src/CMakeFiles/llama.dir"
    Invoke-Checked clang-cl @('/nologo', '/O2', '/MD', '/std:c++17', '/EHsc', "-I$src/src",
        "$PSScriptRoot/test_expert_io.cpp", "/Fo$Out/test_expert_io.obj", '/link', "/OUT:$Out/test_expert_io.exe",
        "$objs/llama-mmap.cpp.obj", "$objs/llama-impl.cpp.obj", "$build/src/llama.lib",
        "$build/ggml/src/ggml-base.lib", "$build/ggml/src/ggml.lib")
    Invoke-Checked "$Out/test_expert_io.exe" @("$PSScriptRoot/../../tests/fixtures/ds41/vq/v3_12b.blob")
}
