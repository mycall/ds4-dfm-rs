**Unsloth UD-IQ2_XXS appears to preserve substantially more of DeepSeek V4.1 Flash 1017's original capabilities than Baekpica's iVQ.** The published evidence is stronger for Unsloth, particularly for reasoning and agentic coding.

Both developers have published quantization fidelity measurements, although they use different evaluation datasets.

## **1. Quantization fidelity**

| Metric                             | Baekpica iVQ       | Unsloth UD-IQ2_XXS   |
| ---------------------------------- | ------------------ | -------------------- |
| Next-token agreement, general text | 78.1% (WikiText-2) | **90.5%** (wiki1100) |
| Next-token agreement, long context | 84.3%              | **91.2%**            |
| KL divergence, general text        | 0.487              | **0.203**            |
| KL divergence, long context        | 0.287              | **0.142**            |
| Calibration-aware                  | Yes                | Yes                  |
| Published task-level accuracy      | Not established    | Yes                  |

Sources: [Baekpica iVQ evaluation](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF?utm_source=chatgpt.com) and [Unsloth GGUF benchmarks](https://unsloth.ai/docs/models/deepseek-v4.1-flash?utm_source=chatgpt.com#gguf-kld-and-top-1-benchmarks).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face+1



Lower KL divergence is better. These figures suggest a meaningful fidelity advantage for Unsloth, but they are not a controlled head-to-head test. Baekpica's initial evaluation also excludes quantization of the non-expert weights.

## **2. Actual reasoning and coding accuracy**

Unsloth's November 2 Dynamic v3.1 evaluation reports the following accuracy retention relative to the original FP4/FP8 checkpoint.

### **UD-IQ2_XXS — original model performance retained**

AIME 2025 — Mathematics

100.00%

GPQA Diamond — Scientific reasoning

99.53%

LiveCodeBench v6 — Programming

100.47%

Combined benchmark recovery

100.00%

Combined 95% confidence interval: 99.40–100.58%. These are relative recovery figures, not absolute benchmark accuracy.

[image](https://www.google.com/s2/favicons?domain=https://unsloth.ai\&sz=32)

Unsloth - Train and Run Models Locally

This is very impressive for a 2-bit quantization. The differences from the original model on these three benchmarks are not statistically significant in Unsloth's reported evaluation.

Baekpica has not published comparable task-completion scores for iVQ, so we cannot establish how much actual reasoning or coding ability it retains.

## **3. Does the newer hVQ close the gap?**

Baekpica's November 11 **hVQ** release uses Hessian-aware vector quantization, improving several fidelity metrics without increasing the model size.

Results measured using the same reference-forward test:

| Metric                         | VQ     | iVQ    | hVQ        |
| ------------------------------ | ------ | ------ | ---------- |
| Next-token agreement           | 82.42% | 82.41% | **84.67%** |
| KL divergence                  | 0.370  | 0.352  | **0.273**  |
| Expert routing agreement       | 75.2%  | 76.5%  | **82.2%**  |
| Hidden state cosine similarity | 0.778  | 0.798  | **0.869**  |
| Expert output error            | 0.309  | 0.288  | **0.189**  |

For coding-related tasks, the improvements are particularly interesting:

| Next-token agreement    | iVQ   | hVQ       |
| ----------------------- | ----- | --------- |
| Python / C code         | 96.4% | **97.0%** |
| Mathematical reasoning  | 93.8% | **95.4%** |
| General reasoning       | 88.4% | **91.6%** |
| Competitive programming | 80.5% | **85.7%** |
| Tool calling / agents   | 71.3% | **76.5%** |
| SWE agentless           | 72.2% | **75.0%** |

These measure token prediction fidelity, not successful task completion. Additionally, hVQ slightly worsens generated-token negative log-likelihood despite improving KL divergence.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face



## **4. The real reason for the accuracy difference**

The weight allocation is substantially different.

### **Model storage allocation**

Backbone

Engram

Unsloth UD-IQ2_XXS

300.5 GB

165.3 GB

135.2 GB

Baekpica iVQ / hVQ

316.6 GB

113.6 GB

203.1 GB

Approximate decimal GB. Excludes separate vision files and external speculative drafters. Baekpica's backbone includes its embedded MTP.

| Specification                | Unsloth UD-IQ2_XXS | Baekpica iVQ/hVQ |
| ---------------------------- | ------------------ | ---------------- |
| Backbone                     | \~165.3 GB         | 113.56 GB        |
| Engram                       | 135.2 GB           | 203.07 GB        |
| Engram precision             | Q5_K               | Original MXFP8   |
| Backbone fits in 128GB       | No                 | Potentially      |
| Expert SSD offloading needed | Yes                | Much less likely |

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

unsloth/DeepSeek-V4.1-Flash-1017-GGUF at main+2

Baekpica allocates more precision to the Engram tables and substantially less to the transformer backbone. Unsloth's backbone has approximately **46% more weight storage**, despite having a smaller total download.

This explains much of Baekpica's potential inference-speed advantage on a single DGX Spark.

## **5. My recommendation for your N1X**

| Configuration              | Quality expectation      | Decode speed            |
| -------------------------- | ------------------------ | ----------------------- |
| Delta + Unsloth UD-IQ2_XXS | Best-supported accuracy  | 10–30 t/s¹              |
| ds4-dfm-rs + iVQ           | Lower expected fidelity  | Not separately measured |
| ds4-dfm-rs + hVQ           | Better fidelity than iVQ | 30–36 t/s²              |

¹ Your existing N1X measurements.

² Published DGX Spark measurements.

**I would keep Unsloth UD-IQ2_XXS for serious agentic software engineering.** I would experiment with hVQ, but skip iVQ entirely.

The fidelity differences are large enough that I would not sacrifice Unsloth's demonstrated accuracy merely to gain another 10–20 tokens per second. A faster model that makes more mistakes, loops during reasoning or generates incorrect tool calls can ultimately take longer to complete a task.

What would change my recommendation is a successful head-to-head comparison of hVQ and UD-IQ2_XXS on DeepSWE, Terminal-Bench or your own OpenCode workloads. I couldn't find such a comparison yet.