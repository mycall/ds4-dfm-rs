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

The two policies returned the same completion text on every turn at each budget. To repeat, set `LLAMA_ARG_CACHE_RAM` to the tested budget and `DS4_N1X_KV_POLICY` to `oldest` or `adaptive` before starting a fresh N1X server. With the model and DSpark configuration above, run one matching check on that fresh server.

```powershell
# 12 MiB, adaptive
python native/windows/smoke-http.py --expect-snapshot-restores 4
# 12 MiB, oldest
python native/windows/smoke-http.py --expect-snapshot-restores 3
# 512 MiB, either policy
python native/windows/smoke-http.py --expect-snapshot-restores 8
```

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
