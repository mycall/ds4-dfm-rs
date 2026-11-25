# Called by build.ps1 inside its native ARM64 VS environment.
param([Parameter(Mandatory)][string]$Out, [switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
$pin = '244640b7f26e2a7301b27a24edc4f7feaf97d52f'
$src = Join-Path $Out 'llama-src'
if (!(Test-Path "$src/.git")) {
    Invoke-Checked git @('init', $src)
    Invoke-Checked git @('-C', $src, 'fetch', '--depth=1', 'https://github.com/ggml-org/llama.cpp.git', $pin)
    Invoke-Checked git @('-C', $src, '-c', 'core.longpaths=true', 'checkout', '--detach', 'FETCH_HEAD')
}
if ((& git -C $src rev-parse HEAD) -ne $pin) { throw "Expected llama.cpp $pin in $src" }
foreach ($name in @('ds41-demand-map.patch', 'ds41-http.patch')) {
    $patch = Join-Path $PSScriptRoot $name
    & git -C $src apply --reverse --check $patch 2>$null
    if ($LASTEXITCODE) { Invoke-Checked git @('-C', $src, 'apply', $patch) }
}
$cuda = if ($CpuOnly) { 'OFF' } else { 'ON' }
$build = Join-Path $Out $(if ($CpuOnly) { 'llama-build' } else { 'llama-cuda' })
$llvm = Join-Path $env:VCINSTALLDIR 'Tools/Llvm/ARM64/bin'
$cmakeTools = Join-Path $env:VSINSTALLDIR 'Common7/IDE/CommonExtensions/Microsoft/CMake'
$env:Path = "$llvm;$cmakeTools/CMake/bin;$cmakeTools/Ninja;$env:Path"
$options = @('-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=clang-cl',
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
