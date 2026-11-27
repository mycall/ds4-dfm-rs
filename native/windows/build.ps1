# Build and smoke-test the native N1X Rust hosts in target/n1x.
# Requires ARM64 MSVC, Windows SDK, VS LLVM, vcpkg, Rust and the ARM64 CUDA Toolkit.
[CmdletBinding()]
param(
    [switch]$Smoke,
    [switch]$VarloopOnly,
    [switch]$GgufCpuOnly,
    [string]$UnslothModel,
    # Explicitly reuse a previously compiled set of native objects.
    [switch]$Relink
)
$ErrorActionPreference = 'Stop'
function Invoke-Checked($tool, [string[]]$arguments) {
    & $tool @arguments
    if ($LASTEXITCODE -ne 0) { throw "$tool exited with $LASTEXITCODE" }
}

if ($UnslothModel) {
    $UnslothModel = (Resolve-Path -LiteralPath $UnslothModel).Path
    $Smoke = $true
}

Push-Location (Join-Path $PSScriptRoot '../..')
try {
    $target = 'aarch64-pc-windows-msvc'
    if ($env:VSCMD_ARG_HOST_ARCH -ne 'arm64' -or $env:VSCMD_ARG_TGT_ARCH -ne 'arm64') {
        $vswhere = Join-Path ([Environment]::GetFolderPath('ProgramFilesX86')) 'Microsoft Visual Studio/Installer/vswhere.exe'
        $install = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
        if (!$install) { throw 'Install the Visual Studio ARM64 C++ tools.' }
        $setup = Join-Path $install 'VC/Auxiliary/Build/vcvarsarm64.bat'
        $vars = & cmd.exe /d /c ('call "{0}" >nul && set' -f $setup)
        if ($LASTEXITCODE) { throw 'ARM64 compiler environment setup failed.' }
        foreach ($line in $vars) {
            if ($line -match '^([^=]+)=(.*)$') {
                [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
            }
        }
    }
    $clang = Join-Path $env:VCINSTALLDIR 'Tools/Llvm/ARM64/bin/clang.exe'
    $vcpkg = Join-Path $env:VCINSTALLDIR 'vcpkg/vcpkg.exe'
    $nvcc = Join-Path $env:CUDA_PATH 'bin/nvcc.exe'
    $env:PATH = (Join-Path $env:CUDA_PATH 'bin/arm64') + ';' + $env:PATH
    $cudaLib = Join-Path $env:CUDA_PATH 'lib/arm64'
    $out = (New-Item -ItemType Directory -Force target/n1x).FullName
    $installed = Join-Path $out 'deps/vcpkg_installed'
    if (!$VarloopOnly) {
        Invoke-Checked $vcpkg @('install', '--triplet=arm64-windows', '--host-triplet=arm64-windows',
                                "--x-manifest-root=$PSScriptRoot", "--x-install-root=$installed")
    }
    $dep = Join-Path $installed 'arm64-windows'
    $common = @('-D_CRT_SECURE_NO_WARNINGS', '-D_CRT_NONSTDC_NO_WARNINGS',
                '-DWIN32_LEAN_AND_MEAN', '-DNOMINMAX', '-D_WIN32_WINNT=0x0A00',
                '-Inative/windows/include', "-I$dep/include", "-I$dep/include/mman")
    $cflags = @('-O3', '-mcpu=native', '-std=gnu17', '-ffunction-sections', '-fdata-sections',
                '-ffast-math', '-fno-finite-math-only', '-fms-runtime-lib=dll', '-I.') + $common
    $gpuFlags = @('-O3', '-gencode=arch=compute_121a,code=sm_121a', '--use_fast_math', '-DDS4_CUDA_HAVE_MXF4=1',
                  '-std=c++20', '-Xcompiler=/O2,/MD,/Zc:preprocessor,/bigobj', '-Icuda/mmq') + $common
    # This binary needs only the Rust host and its own CUDA projections.
    $env:CMAKE_TOOLCHAIN_FILE_aarch64_pc_windows_msvc = Join-Path $PSScriptRoot 'runtime.cmake'
    $env:CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_RUSTFLAGS = '-C target-cpu=native -C target-feature=-crt-static'
    $env:CARGO_PROFILE_RELEASE_LTO = 'fat'
    $env:CARGO_PROFILE_RELEASE_CODEGEN_UNITS = '1'
    if ($VarloopOnly) {
        $varloop = Join-Path $out 'varloop.obj'
        Invoke-Checked $nvcc ($gpuFlags + @('-c', 'cuda/varloop.cu', '-o', $varloop))
        $varloopHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $varloop).Hash
        Invoke-Checked cargo @('rustc', '--release', '--locked', '--target', $target,
            '-p', 'ds4-cli', '--bin', 'ds4-varloop', '--features', 'varloop-cuda', '--',
            '-C', "metadata=$varloopHash", '-C', "link-arg=$varloop",
            '-C', "link-arg=/LIBPATH:$cudaLib", '-C', 'link-arg=cudart.lib')
        Copy-Item "target/$target/release/ds4-varloop.exe" $out
        if ($Smoke) {
            $src = Join-Path $out 'llama-src'
            $reference = Join-Path $out 'llama-cuda'
            Invoke-Checked cl.exe @('/nologo', '/O2', '/MD', '/EHsc', '/std:c++17',
                "/I$src/include", "/I$src/ggml/include", "$PSScriptRoot/varloop_oracle.cpp",
                "/Fo$out/varloop_oracle.obj", '/link', "/OUT:$out/varloop_oracle.exe",
                "$reference/src/llama.lib")
        }
        return
    }
    $cSources = @('ds4.c', 'ds4_ple.c', 'native/bridge/ds4_bridge.c', 'native/windows/single_host.c')
    $gpuSources = @('ds4_cuda.cu', 'cuda/qwen38_ple.cu', 'cuda/mmq/ds4_ggml_stubs.cu',
                    'cuda/mmq/ds4_mmq.cu', 'cuda/mmq/ds4_mmq_d2r.cu', 'cuda/mmq/quantize.cu',
                    'cuda/mmq/mmid.cu', 'cuda/mmq/mmvq.cu', 'cuda/mmq/ds4_repack.cu', 'cuda/mmq/ds4_fattn.cu')
    $objects = foreach ($source in ($cSources + $gpuSources)) {
        $obj = Join-Path $out ([IO.Path]::GetFileNameWithoutExtension($source) + '.obj')
        if (!$Relink) {
            Write-Host "Compiling $source"
            if ($source.EndsWith('.cu')) {
                & $nvcc @gpuFlags -c $source -o $obj | Out-Host
            } else {
                & $clang @cflags -c $source -o $obj | Out-Host
            }
            if ($LASTEXITCODE) { throw "Compilation failed for $source" }
        }
        if (!(Test-Path -LiteralPath $obj)) { throw "Missing $obj" }
        $obj
    }
    & "$PSScriptRoot/build-gguf.ps1" -Out $out -CpuOnly:$GgufCpuOnly -Smoke:$Smoke
    $libdirs = @("/LIBPATH:$cudaLib", "/LIBPATH:$dep/lib")
    $libs = @('cuda.lib', 'cublas.lib', 'cudart.lib', 'pthreadVC3.lib', 'mman.lib', 'OneCore.lib')
    $link = foreach ($arg in ($objects + $libdirs + $libs)) { '-C'; "link-arg=$arg" }
    $fingerprint = (Get-FileHash -Algorithm SHA256 -LiteralPath $objects).Hash -join ''
    # Cargo otherwise reuses a binary when only the native objects change.
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $hash = [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::ASCII.GetBytes($fingerprint))).Replace('-', '')
    } finally { $sha.Dispose() }
    foreach ($entry in @(@('ds4-server', 'ds4-server-rs'), @('ds4-cli', 'ds4-rs'), @('ds4-cli', 'ds4-bench-rs'))) {
        $pkg, $bin = $entry
        $features = if ($bin -eq 'ds4-rs') { 'n1x-gguf' } else { 'native' }
        $extra = if ($bin -eq 'ds4-rs') { @('-C', "link-arg=$out/ds41_gguf.lib") } else { @() }
        Invoke-Checked cargo (@('rustc', '--release', '--locked', '--target', $target, '--features', $features,
                                '-p', $pkg, '--bin', $bin, '--', '-C', "metadata=$hash") + $link + $extra)
        Copy-Item "target/$target/release/$bin.exe" (Join-Path $out ($bin.Replace('-rs', '') + '.exe'))
    }
    Copy-Item "$dep/bin/mman.dll","$dep/bin/pthreadVC3.dll" $out
    if ($Smoke) {
        Invoke-Checked $clang ($cflags + @('tests/test_shard_map.c', '-o', "$out/test_shard_map.exe",
                                           '-Xlinker', 'OneCore.lib'))
        Invoke-Checked "$out/test_shard_map.exe" @('tests/fixtures/ds41/vq/v2_11b.blob',
                                                   'tests/fixtures/ds41/vq/v3_13b.blob')
        Invoke-Checked "$out/ds4.exe" @('--help')
        Invoke-Checked "$out/ds4-server.exe" @('--version')
        Invoke-Checked "$out/ds4-bench.exe" @('--help')
        Invoke-Checked $nvcc ($gpuFlags + @('-c', 'tests/test_ds41_vq.cu', '-o', "$out/test_ds41_vq.obj"))
        Invoke-Checked link.exe (@('/NOLOGO', '/OPT:REF', "/OUT:$out/test_ds41_vq.exe",
                                   "$out/test_ds41_vq.obj") + $objects + $libdirs + $libs)
        foreach ($fixture in @('v3_13b', 'v3_12b', 'v2_12b', 'v2_11b')) {
            $base = "tests/fixtures/ds41/vq/$fixture"
            $inputs = @("$base.blob", "$base.probes.txt")
            if ($fixture.StartsWith('v3')) { $inputs += "$base.ref.f32" }
            Invoke-Checked "$out/test_ds41_vq.exe" $inputs
        }
        if ($UnslothModel) {
            $probe = "$out/test_ds41_gguf_expert"
            Invoke-Checked $clang ($cflags + @("-I$env:CUDA_PATH/include", '-c',
                                               'tests/test_ds41_gguf_expert.c', '-o', "$probe.obj"))
            # The probe includes the native loader and scalar reference from ds4.c.
            $support = @($objects | Where-Object { [IO.Path]::GetFileName($_) -ne 'ds4.obj' })
            Invoke-Checked link.exe (@('/NOLOGO', '/OPT:REF', "/OUT:$probe.exe", "$probe.obj") +
                                     $support + $libdirs + $libs)
            Invoke-Checked "$probe.exe" @($UnslothModel)
            $backend = if ($GgufCpuOnly) { 'cpu' } else { 'cuda' }
            Invoke-Checked python @("$PSScriptRoot/smoke-gguf.py", "$out/ds4.exe", $UnslothModel, '--backend', $backend)
        }
    }
} finally {
    Pop-Location
}
