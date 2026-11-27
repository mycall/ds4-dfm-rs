# DeepSeek V4.1 DFly2 benchmark on Windows N1X

This single-sequence native benchmark loads AngelSlim's four original safetensors into the N1X CUDA backend. It compares greedy decoding with DFly2-B6, the DSpark GGUF sidecar, and the target alone. It runs against the repository's Windows GGUF backend and Rust expert-cache policy. The HTTP server has its own entry point.

The DFly2 graph reads the target's final normalized hidden state and the window keys and sparse selections from layers 28, 32 and 36. It uses the compressed keys from layer 20, the target embedding and the target output head. Its three MoE layers produce a block of six slots. The anchor occupies slot zero. The five subsequent token proposals each use attentive correction conditioned on the preceding token. The target scores the proposals and rolls back at the first disagreement. The implementation follows the [AngelSlim checkpoint](https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6) and [Tencent's AngelSpec reference](https://github.com/Tencent/AngelSpec/tree/e02ea36e65d44e6e165c4ba73c139d500e12f2d9/angelspec/models/draft).

This implementation keeps a host copy of the target intermediates, assembles the selected rows on the CPU, and builds a fresh GGML draft graph per round. The dense draft weights use BF16 or F32. The loader preserves each FP4 expert's codes and group scale in the native MXFP4 layout. Attention uses F16 keys and values, and GGML controls accumulation and intermediate precision. Upstream BF16 proposal-logit parity and stochastic sampling are outside the measurements below.

## Measurement setup

The November 27, 2026 runs used the 128 GiB RTX Spark N1X, Windows ARM64, driver 621.24 and CUDA 13.5. The starting repository commit was `d600479d6ad7b8fdfd4ea5c35eeb7329a08d9727` and the Windows backend's llama.cpp pin was `244640b7f26e2a7301b27a24edc4f7feaf97d52f`. The six-shard Unsloth UD-IQ2_XXS target totaled 279.84 GiB on disk. All modes used direct expert I/O, learned cache seeding, disabled expert prefetch, and the same 68 GiB expert cache. The short raw-completion prompts contained 29 and 33 tokens. The 1,846-token prompt included the repository's Windows HTTP launcher source and a question about its cache policy.

Warm decode rates below show the range of passes two and three, in tokens per second. Pass one used a fresh process for each configuration. The CSV preserves all 36 measurements, including first-pass results and phase timings. [Per-run measurements](results-n1x-2026-11-27.csv)

| Prompt | Plain | DSpark, five proposals | DFly2, five proposals | DFly2, two proposals |
|---|---:|---:|---:|---:|
| Write a Python function | 11.15–12.20 | 15.35–15.60 | 12.11–12.67 | 11.12–11.17 |
| Explain a write-ahead log | 10.65–12.35 | 9.05–9.06 | 7.45–7.47 | 9.64–9.73 |
| Answer from launcher source | 11.21–12.45 | 11.34–11.86 | 7.67–7.85 | 8.46–8.92 |

DFly2 with five proposals accepted 50/61 tokens on code, 36/126 on prose, and 38/125 on the source question. DSpark accepted 51/56, 38/122, and 41/104. Their verified 64-token continuations matched each other on all three prompts. Plain decoding first differed at token seven on prose and token six on the source question. These are timings of the naturally generated continuations.

DFly2 spent 49–51 ms drafting per short-prompt round and 71–73 ms on the source question with five proposals. DSpark spent 7.6–7.9 ms on the same runs. If DFly2 drafting took zero time and its measured verifier cost and acceptance stayed fixed, its warm decode ceiling would be 8.9 tok/s on prose and 9.8–10.2 tok/s on the source question. Reducing proposal depth to two helped both cases. Plain decoding was faster in both. The code-prompt ranges for plain and five-proposal DFly2 overlap.

A fresh build of the preserved source repeated the five-proposal test and reproduced all 192 generated IDs and the acceptance counts from the timing run. In another fresh process, the target alone replayed the proposal blocks and reproduced all 192 IDs and all three acceptance counts. The four drafter shards totaled 8,847,525,344 bytes. During the DFly2 repeat, one-second `nvidia-smi` samples peaked at 92,125 MiB of whole-GPU usage out of 98,304 MiB dedicated. The target streamed offloaded experts from its GGUF shards.

## Reproduce

Start with the repository's native ARM64 N1X CUDA GGUF build in `target/n1x`. Run the following from the repository root in PowerShell 7. Set the three model variables to the local artifact paths. The build puts the probe and its private target-output DLL under `target/n1x-dfly2-bench`.

```powershell
$bench = './native/windows/dfly2-bench'
$models = "$HOME/models/deepseek-ai/DeepSeek-V4.1-Flash-1017"
$target = (Get-ChildItem "$models/unsloth/UD-IQ2_XXS/*-00001-of-*.gguf").FullName
$angel = "$HOME/models/AngelSlim/DS-V4.1-Flash-DFly2-B6"
$dspark = "$models/experimental/DeepSeek-V4.1-Flash-1017-DSpark-IQ2_XXS-imx-dflash.gguf"
& "$bench/build.ps1"
& "$bench/run.ps1" -Mode dfly2 -Target $target -Draft $angel
& "$bench/run.ps1" -Mode dspark -Target $target -Draft $dspark
& "$bench/run.ps1" -Mode plain -Target $target
```

To compare the launcher's one-shot arithmetic prompt on the looped target, use the same model variables as above. `-Prompt` uses the embedded chat template, the N1X CLI's default system message, greedy decoding and the first end-of-generation token. The two extra visits to blocks 15–18 give 48 logical layers. Both drafters read the final visit to each checkpoint layer they require. With this loop, DFly2 reads compressed layer 28 and attention layers 36, 40 and 44. DSpark reads layers 45–47. The plain arm measures the same verifier one token at a time.

```powershell
$env:DS4_N1X_LOOP_LAYERS = '15-18'
$env:DS4_N1X_LOOP_REPEATS = '2'
try {
    $test = @{ Target = $target; Prompt = 'What is two plus two?'; Tokens = 8; Context = 512; CacheMiB = 49184; Repeats = 4 }
    & "$bench/run.ps1" @test -Mode dfly2 -Draft $angel
    & "$bench/run.ps1" @test -Mode dspark -Draft $dspark
    & "$bench/run.ps1" @test -Mode plain
} finally {
    Remove-Item Env:DS4_N1X_LOOP_LAYERS, Env:DS4_N1X_LOOP_REPEATS
}
```

On November 27, 2026, all three modes returned `Two plus two is **4**.` in seven visible tokens followed by EOS. The Rust tokenizer reproduced all 16 prompt IDs. Target-only replay reproduced DFly2's eight output IDs and acceptance count on every pass. Each mode ran in a fresh process, followed by three repeats of the identical prompt. Request time covers prefill and decode after model loading.

| 48-layer target mode | Warm decode range (tok/s) | Median warm request | Accepted / proposed per request |
|---|---:|---:|---:|
| Plain | 16.81–17.26 | 0.682 s | — |
| DSpark IQ2_XXS | 28.28–31.23 | 0.563 s | 5 / 5 |
| DFly2-B6 | 8.34–10.41 | 1.011 s | 3 / 11 |

### 32K retest on the 48-layer target

The same loop, cache, eight-token budget and five-proposal limit ran with `-Context 32768`. On the original 16-token arithmetic prompt, median warm decode was 16.91 tok/s for plain, 30.03 for DSpark and 10.46 for DFly2. A second fixture placed the question after repository source. Each mode processed all 32,752 rendered input tokens once in a fresh process. All three generated the same eight IDs, including EOS. Times exclude model loading. [All 15 measurements](results-n1x-loop48-32k-2026-11-27.csv)

| Mode | Long prefill | Decode after first token | Decode tok/s | Accepted / proposed | Sampled peak GPU use |
|---|---:|---:|---:|---:|---:|
| Plain | 1,122.87 s | 2.28 s | 3.07 | — | 61.63 GiB |
| DSpark IQ2_XXS | 1,108.11 s | 3.58 s | 1.96 | 4 / 10 | 66.27 GiB |
| DFly2-B6 | 1,122.30 s | 3.58 s | 1.96 | 4 / 10 | 69.96 GiB |

DFly2 spent 0.27 s drafting and 3.30 s verifying. DSpark spent 0.05 s drafting and 3.53 s verifying. Prefill accounted for over 99.6% of each full-length request. The single long run per mode leaves the prefill repeat range unmeasured.

Each invocation starts a fresh process. The default is three passes over the code, prose and source-file prompts, with a 68 GiB expert cache, 32,768-token context and 64 generated tokens. The first pass warms the expert cache. Each request clears the model's sequence memory. Decode throughput counts the 63 tokens after the first token and excludes prefill and model load. Each JSONL row includes the prompt, generated IDs, prefill time, decode time, draft time, verifier time, candidates, and accepted count. The probe measures fixed-length greedy continuations, including any EOS token within that budget.

Use `-DraftMax 2` to verify up to two proposals per round. DFly2 still processes all six backbone slots because they attend to one another. It runs the sequential correction and output head only for the requested proposal slots.

For a correctness check, pass a DFly2 or DSpark result file to the target alone. Replay uses the same proposal blocks, recomputes the acceptance decisions, and fails on any difference in the generated token IDs. Use the original token budget if it differs from 64.

```powershell
$trace = Get-ChildItem './target/n1x-dfly2-bench/dfly2-64-*.jsonl' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
& "$bench/run.ps1" -Mode replay -Target $target -ReplayLog $trace.FullName
```
