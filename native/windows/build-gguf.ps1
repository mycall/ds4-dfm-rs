# Called by build.ps1 inside its native ARM64 VS environment.
param([Parameter(Mandatory)][string]$Out, [switch]$CpuOnly, [switch]$Smoke)
$ErrorActionPreference = 'Stop'
$src = Join-Path $Out 'llama-src'
& "$PSScriptRoot/update-llama-source.ps1" -Src $src
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
