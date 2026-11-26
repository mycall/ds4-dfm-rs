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
