# N1X expert seeding and attention KV eviction

Windows ARM64 RTX Spark N1X, CUDA 13.5.37, driver 621.24. The Windows GGUF server used the six-shard Unsloth DeepSeek-V4.1-Flash-1017 UD-IQ2_XXS, the IQ2_XXS imatrix DSpark sidecar at depth five, one 32,768-token slot and the existing host clock policy. Measurements were made on the November 25 local checkout of `n1x` based on `767a9191`, with llama.cpp pinned at `244640b7`.

## Expert weights

Four coding requests supplied the training routes. For each compatible bank, rank the experts by the number of batches that used them. A layer's quota is its number of entries among the winners. The total initial seed allowance in each bank equals its original equal-share allowance. The runtime checks the architecture, routed expert count, group layout, per-expert strides and slot counts across all 40 layers before selecting these quotas. A mismatch uses equal quotas. Ordinary decode replacement continues to use the shared ClassLRU.

The evaluation used the six allocation requests in order with 150 generated fixture functions, 40 host expert layers, 75,264 MiB of expert cache, 512 MiB of snapshot RAM and two outstanding expert reads. Each run started a fresh server. Wall times cover the HTTP requests after readiness. Read bytes and read operations are Windows process I/O deltas during the requests. These include sector padding and other process reads. Storage-controller traffic was not measured. [Delta fixture at 5f9a130d](https://github.com/mycall/delta-inference-engine/blob/5f9a130da793733d38a5c989d35f4d2e09d2a1a6/tools/v41/bench_allocation.py)

| Seed policy | Read GiB | Read operations | Six requests, seconds | Server prefill, seconds | Server decode, seconds |
|---|---:|---:|---:|---:|---:|
| Equal | 165.49 | 44,457 | 99.40–105.62 | 88.36–93.77 | 10.88–11.63 |
| Learned | 163.96 | 44,033 | 98.53–105.44 | 87.61–93.81 | 10.68–11.41 |
| Two-stage | 151.00 | 41,600 | 97.07–100.17 | 87.18–89.51 | 9.64–10.38 |

Equal includes the pre-change binary and the integrated `equal` setting. The byte totals reproduced across both runs of each policy. All six returned the same text, prompt and completion token counts, reused prefix lengths and draft acceptance across policies. Five passed the strict JSON fixture. The warm-up consistently returned an object containing 4 when the fixture requested the scalar 4.

The learned wide-batch policy lowers aggregate read volume without increasing it on any request in this fixture. Two-stage seeding also acts on a 33–127-token batch. It saves more bytes in aggregate. Its first long prompt read 53.63 GiB, down from 66.86 GiB, and its second long prompt read 37.32 GiB, up from 33.02 GiB. The operator reported browser activity during an earlier timing comparison. Times reflect a shared workstation with normal filesystem caches. Busy-clock medians during these runs ranged from 1,677 to 1,800 MHz, and available host RAM stayed above 9.77 GiB. During the second long prompt, sampled browser processes used 0.03–0.13 CPU-seconds in the four later runs with browser telemetry. Browser GPU and storage traffic were not isolated. Read volume gives a more repeatable comparison here than wall time.

## Attention KV snapshots

Adaptive eviction chooses between the oldest recent snapshot and the oldest frequently restored snapshot. Evicted text histories leave a SHA-256 fingerprint, size and class. A return to an exact evicted history adjusts the byte target for the recent class. The history holds at most 256 fingerprints with a combined reference size bounded by the RAM budget. Snapshots with adapters or media still use the original compatibility and restore checks. They leave no fingerprint. Payloads include target state, draft state and checkpoints. An oversized snapshot still skips admission.

The HTTP regression alternates six distinct histories of 32–72 tokens in the order `ABACADAEBFAECA`. Each completion requests one greedy token. A hit is visible on the wire because distinct histories only share BOS. The server log independently identifies each successful snapshot restore.

| Snapshot budget | Oldest restores | Adaptive restores | Oldest reused tokens | Adaptive reused tokens |
|---|---:|---:|---:|---:|
| 12 MiB | 3 | 4 | 84 | 112 |
| 512 MiB | 8 | 8 | 280 | 280 |

The two policies returned the same completion text on every turn at each budget. To repeat, set `LLAMA_ARG_CACHE_RAM` to the tested budget and `DS4_N1X_KV_POLICY` to `oldest` or `adaptive` before starting a fresh N1X server with `-MaxSeqs 1 -Context 32768`. With the model and DSpark configuration above, run one matching check on that fresh server.

```powershell
# 12 MiB, adaptive
python native/windows/smoke-http.py --expect-snapshot-restores 4
# 12 MiB, oldest
python native/windows/smoke-http.py --expect-snapshot-restores 3
# 512 MiB, either policy
python native/windows/smoke-http.py --expect-snapshot-restores 8
```

## Four concurrent conversations

On November 25, the four-slot Windows host based on `cdc6ba5b` used the same six target shards, automatic expert-cache sizing, no drafter, and no snapshot RAM. The server opened four slots of 1,048,576 tokens each with a 512-row CUDA microbatch. A 2,048-row microbatch at that capacity had failed to allocate a 45.8 GiB prefill buffer.

`python native/windows/smoke-http.py --concurrent --parent-lines 2080` sent a long parent prompt, three unrelated chats, then four simultaneous follow-ups to the same resident server. Times measure the first nonempty streamed content.

| Parent request | Prompt tokens | Cached tokens | Server prefill | Time to first content |
|---|---:|---:|---:|---:|
| Initial | 26,080 | 1 | 432.94 s | 433.00 s |
| After the three other chats | 26,097 | 26,082 | 2.03 s | 2.19 s |

All four follow-ups overlapped, preserved their own private words and processed 15 new prompt tokens each. Their times to first content ranged from 1.73 to 2.19 s. During the long prefill, two Windows samples on the model drive showed 2.53 and 3.16 GB/s of physical reads, 51% and 58% disk active time, and 2.60 and 3.01 GB/s of model-process reads, respectively. These are samples from one run. The integration test checks reuse, response isolation and overlap without machine-specific latency thresholds.

## Read volume at the 250,000-token four-slot default

On November 26, fresh N1X CUDA HTTP servers ran the same six-shard model with four 250,000-token slots, automatic expert-cache sizing, direct expert reads and no drafter. Each server processed the same 8,192-token repository excerpt from an empty prompt cache in 2,048-row microbatches and generated the same single token. Windows process read counters were sampled after readiness and after the response. The physical disk counters agreed with the process totals to within 0.02 GB. Units below are decimal.

| Seed policy | Process read GB | MB per prompt token | Read operations | Server prefill | Prompt throughput |
|---|---:|---:|---:|---:|---:|
| Disabled | 478.04 | 58.35 | 48,970 | 103.96 s | 78.8 tok/s |
| Equal, production default | 332.82 | 40.63 | 57,343 | 86.44 s | 94.8 tok/s |
| Experimental rolling refresh | 330.78 | 40.38 | 58,105 | 86.72 s | 94.5 tok/s |

Equal seeding saved 145.22 GB (30.4%) using the existing GPU allocation. A second, instrumented equal run read the same number of bytes and took 88.96 s. The rolling prototype updated per-expert routing counts after each wide microbatch, weighting the previous score by 0.75. It saved another 2.04 GB on this prompt but issued 762 more reads. At four 1M-token slots (512-row microbatches), rolling saved 7.4 GB and 21.2 GB on two successive repository excerpts, then increased reads on their cached-prefix generation request from 13.20 to 13.77 GB. Equal seeding remains the default.

With four 1M-token slots, the 3,378-token excerpt alone read 654.67 GB with seeding disabled and 385.74 GB with equal seeding. A separate four-slot integration run at that capacity took 131.87 and 95.65 seconds respectively and returned the digit 4 in both cases.

Replaying expert selection for 8,188 wide-batch tokens from the 8,192-token prompt estimated 40.55 MB/token at 2,048 rows, 26.26 at 4,096 and 17.45 at 8,192. The two larger widths are estimates of expert payload only. GPU memory fit and elapsed time at those widths remain unmeasured.

To check the default on the local model, run the integration test with the server stopped. It launches a fresh server for each policy, sends the same multi-batch prompt, and checks the answer and the reduction in process read bytes.

```powershell
.\native\windows\test-prefill-reads.ps1 -Model $model
```

### Four simultaneous 8,192-token prefills

On `ce9b51fd`, four clients each submitted a different 8,192-token repository excerpt to a fresh server within 13 ms. Each requested one greedy completion token. Both policies ran once with the same input tokens, request order and four-slot settings described above. Every response reported zero cached prompt tokens. Each prompt produced the same completion across policies.

| Seed policy | Server reads | MB per prompt token | All requests complete | Aggregate prompt tok/s | Time to first token by arrival order |
|---|---:|---:|---:|---:|---|
| Disabled | 1,496.68 GB | 45.68 | 364.09 s | 90.0 | 129, 220, 303, 364 s |
| Equal (default) | 1,201.54 GB | 36.67 | 332.66 s | 98.5 | 108, 191, 272, 333 s |

Equal seeding saved 295.14 GB (19.7%) and 31.43 seconds. All four slots were occupied before the first response. In the seeded run, slot 0 had processed 8,188 tokens when slot 1 first reported four. Aggregate throughput uses all 32,768 input tokens divided by elapsed client time. [Input hashes and sources](n1x-four-prefills/inputs.json) · [Default run](n1x-four-prefills/default.json) · [Disabled run](n1x-four-prefills/disabled.json)

## ARM64 host measurements

Windows exposed NEON, dot product, I8MM, BF16, SVE and SVE2 on this N1X. An SVE loop measured a 16-byte vector length. Windows reported SME and SME2 unavailable. The release already builds the quantized CPU backend with `-mcpu=native`. [Windows processor feature detection](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent)

Timers on the host thread attributed 57.80 s to finishing outstanding expert reads and 9.02 s to resolving and submitting those reads in a 103.00 s allocation suite. CPU top-k sampling took 8.48 ms across the same six requests. Eliminating that entire sampling step would have saved at most 0.0083% of the measured wall time.

An experimental filter used the first 512 candidates to derive a lower bound for the 40th-largest logit. It retained every later candidate that could enter the heap. On 16 captured 129,280-entry vocabularies, the existing partial sort took a median 46.29 µs per vocabulary, the NEON filter 32.11 µs and the SVE gather filter 48.02 µs. When large logits appeared only after the initial prefix, the NEON median rose to 283.01 µs against 93.17 µs for the existing implementation.

Four fresh CUDA servers ran the 260-token coding loop in baseline / NEON / NEON / baseline order. The existing sampler spent 34.28–34.82 ms in top-k, and the NEON candidate spent 44.84–45.83 ms. All four returned the same completion and draft acceptance, with request times of 33.87–33.97 s. The release keeps the existing sampler.

The CPU-only control rebuilt all 17 objects in the ggml CPU backend for Armv8-A with NEON. The rest of the runtime used its deployed flags. A first attempt with 32 prompt tokens hit the five-minute watchdog. The completed test used four fixed prompt tokens followed by one teacher-forced decode step, 12 threads and two fresh processes per build. The deployed runs preceded the coherent Armv8-A runs. Each pair repeated its two top-token IDs.

| CPU backend | Four-token prefill, seconds | One decode step, seconds | Top token at each position |
|---|---:|---:|---|
| Deployed native | 53.39–60.66 | 50.16–54.85 | 36651, 2105 |
| Armv8-A NEON | 59.32–61.81 | 58.99–64.33 | 11, 2105 |

The prefill top token changed between backends. Two prediction positions and sequential run order support only a narrow comparison. An earlier mixed build combined a two-row dispatcher with one-row quantized kernels and returned uniform logits. Its timings were excluded.

A separate single-thread IQ2_XXS × Q8_K dot-product test used 2,304 synthetic weight rows of width 5,120. The harness pinned the thread to one logical CPU and warmed it for 800 ms before timing. Per-process median times were 1.463–1.583 ms for the native build and 1.703–1.845 ms for Armv8-A, with matching output checksums. This covers one quantization format and a hot weight buffer.

## Repeated DeepSeek layers and active context

November 26, 2026, native Windows ARM64 N1X server on the 96 GiB device, using the six local DeepSeek-V4.1-Flash-1017 UD-IQ2_XXS shards, no drafter, a 64 GiB GPU expert cache and CPU boost enabled. The launcher allocated four slots of 250,112 tokens each. One slot was active. This checkout was based on `c87b76f5`. The depth and context runs used `llama.dll` SHA-256 `a3da631087364525f6f91a35f29a50b34afcf553009c44a12e90609019413a9c`.

With layers 15–18 repeated once, two fresh starts per cache size generated the same 128-token answer. With separate entries for each visit, automatic sizing gave 7.38–7.42 tokens/s and 53.53 GiB of process reads. A 64 GiB cache gave 8.67–8.91 tokens/s and 38.34 GiB of reads. Peak sampled GPU usage rose from about 71 to 88 GiB. On an immediate repeat of a shorter prompt, both cache sizes decoded at about 19 tokens/s.

At 64 GiB, two fresh starts per sharing setting generated that same 128-token answer at 8.77–8.91 tokens/s with separate entries and 9.37–9.67 tokens/s with shared entries. Sharing reduced process reads from 38.34 to 32.44 GiB.

For the context comparison, each mode started a fresh server and processed the same tokenized repository text at successively longer frontiers. At each frontier a one-token request filled the KV cache. Resending the prompt evaluated the last four prompt tokens and generated 128 greedy tokens. All three pairs returned identical token IDs. SSD bytes are raw Windows PhysicalDisk counter deltas for the model's Samsung 990 PRO, including other reads on that drive.

Windows Kernel-Power events identify source changes during a run. The context comparison uses a fresh run after reconnection. The depth sweep retains the original deeper runs and records their power conditions.

Each pair below gives **separate → shared**. Prefill wall time covers the first request. The final two columns describe the 128-token request with the prompt already in KV.

| Occupied tokens | Newly evaluated during fill | Prefill request wall (s) | Generation request SSD reads (GiB) | Decode (tok/s) |
|---:|---:|---:|---:|---:|
| 1,024 | 1,024 | 30.49 → 28.51 | 22.60 → 19.74 | 9.37 → 10.09 |
| 8,192 | 8,192 | 84.09 → 74.59 | 9.71 → 8.17 | 12.29 → 13.13 |
| 16,384 | 10,240 | 104.18 → 92.07 | 7.79 → 5.89 | 13.47 → 14.05 |

At 16,384 tokens the fill reused 6,144 prompt tokens in both modes and read 321.36 and 281.63 GiB from the drive, respectively. An earlier shared run that included battery power also read 5.89 GiB on generation and decoded at 13.29 tokens/s. Each table cell is one measurement. Expert-cache history and generated text vary across frontiers.

An AC control without a layer repeat used the same prompts, lengths, cache budget and request order. Decode reached 11.09, 13.56 and 15.38 tokens/s at the three frontiers. At 16,384 tokens the ordinary graph read 5.97 GiB for the 128-token request and completed it in 8.41 seconds. The shared loop read 5.89 GiB and took 9.23 seconds. Both produced 128 tokens, with different continuations and potentially different expert routes.

### Extra passes over layers 15–18

Each fresh server processed a 3,378-token prompt, two short generation requests, and a 46-token database question with a 128-token answer. Prompt reuse was disabled for these measurements. All depths then answered the same 12 four-choice ARC questions. For each question, the scorer calculated the negative log probability of the correct label among the four choices. Lower loss means a higher geometric mean probability for the correct answer.

| Extra passes | Logical layers | Prefill (tok/s) | 128-token decode (tok/s) | Process reads for 128-token request (GiB) | ARC correct | Mean four-choice loss | Power |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 40 | 75.17 | 10.53 | 30.82 | 12/12 | 0.00004 | AC rerun |
| 1 | 44 | 68.16 | 9.57 | 32.44 | 12/12 | 0.00026 | AC |
| 2 | 48 | 60.62 | 8.89 | 35.56 | 12/12 | 0.01052 | Reconnected during prefill |
| 3 | 52 | 55.38 | 7.98 | 39.39 | 12/12 | 0.03841 | Battery |

These are single-run rates per depth. All depths produced 128 tokens on the database prompt, and the generated text differed across depths. On the generated GGUF fixture, each depth matched all 64 logits at five frontiers against explicit layer copies on both CPU and CUDA.
