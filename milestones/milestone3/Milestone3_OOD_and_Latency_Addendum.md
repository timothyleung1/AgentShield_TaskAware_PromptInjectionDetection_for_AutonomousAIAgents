# AgentShield — Milestone 3 Addendum: T4 Reproduction, Adapter Merging Latency & External OOD Generalization

**Technical addendum | 25 November 2026**  
**Authors:** Timothy Leung, Eva Liu  
**Scope:** Independent single-epoch LoRA reproduction on NVIDIA Tesla T4 (`float16`), controlled unmerged-vs-merged (`.merge_and_unload()`) inference latency verification, and a 3-way external out-of-distribution (OOD) generalization evaluation comparing `0-shot v2`, frozen `8-shot ICL (random_seed42_8shot)`, and `Merged LoRA` on `Qwen/Qwen2.5-3B-Instruct`.

> **Protocol guardrail:** None of the follow-up experiments below touch, inspect, or tune on the sealed `agentic-prompt-injection-5k` test split (`n = 545`). All in-domain checks use the 467-row validation split only; all new evaluations use separate public datasets or human-written probes.

---

## 1. Independent GCP Tesla T4 Reproduction (1 Epoch, `float16`)

To verify reproducibility across GPU architectures/dtypes and record complete wall-clock training cost and memory metrics, we re-ran `train_lora.py` from scratch on a single **NVIDIA Tesla T4 (16 GB)** in `float16` for 1 full epoch (`491` optimizer steps over all `3,923` training rows, `batch=2`, `grad_accum=4`, `seed=42`, `lr=1e-4`, `r=8`, `alpha=16`, 7 linear target modules):

- **Wall-clock training time:** `2,454.1 s` (**40m 54s**), `1.599 samples/s`
- **Peak GPU memory:** `8.19 GB` allocated (`15.19 GB` reserved)
- **Final validation loss:** `0.00273` (`test_split_accessed: false`)

### In-Domain Validation Reproduction (`n = 467`)

| Candidate | Hardware / Dtype | Accuracy | Precision | Recall | F1 | Benign FPR |
|---|---|---:|---:|---:|---:|---:|
| M1b Base 0-shot v2 | Apple M3 Pro (`mps`) | 0.8822 | 0.9534 | 0.8636 | 0.9063 | 0.0818 |
| M2 8-shot ICL (`s42`) | Apple M3 Pro (`mps`) | 0.9443 | 0.9608 | 0.9545 | 0.9577 | 0.0755 |
| **Independent LoRA (1 ep)** | **Tesla T4 (`float16`)** | **0.9936** | **0.9904** | **1.0000** | **0.9952** | **0.0189** |
| Primary LoRA (`ckpt-491`, 1 ep) | NVIDIA A10G (`bfloat16`) | 0.9979 | 0.9968 | 1.0000 | 0.9984 | 0.0063 |

All four attack families reached **100.0% recall** (`308/308`, `0 FN`) on the T4 reproduction; the 3 benign false positives were `quoting_attack_for_analysis` (`id=655, 3172, 4544`), down from `11/12` under 8-shot ICL.

---

## 2. Engineering Finding: Adapter Merging Eliminates PEFT Latency Overhead

In the primary A10G validation run, unmerged `PeftModel` LoRA ran **62% slower** than 8-shot ICL (`0.4925 s` vs. `0.3035 s` mean latency) despite using **58% fewer input tokens** (`345.7` vs. `828.7`). Code inspection identified the root cause: `LoRADetector` kept `PeftModel` wrappers active across `252` linear layers (`36` transformer layers × `7` modules), executing ~504 extra small adapter matrix multiplications per decoded token at batch size 1.

Benchmarking `.merge_and_unload()` (`W' = W + (alpha/r) B A` folded once at model load) on 30 validation rows on the Tesla T4 (`float16`, batch size `1`):

| Mode | Exact Agreement | Mean Latency | p50 Latency | Speedup |
|---|---:|---:|---:|---:|
| Unmerged `PeftModel` | — | 0.7753 s | 0.7795 s | 1.00× |
| **Merged `.merge_and_unload()`** | **30 / 30** | **0.4036 s** | **0.4039 s** | **1.92×** |

Across the external OOD benchmarks on the same T4, merged LoRA sustained **`0.38–0.43 s` p50** per request — approximately **30% faster** than 8-shot ICL (`0.55–0.59 s` p50) while using **58% fewer prompt tokens**. We added an opt-in `--merge-adapter` flag to [`evaluate_lora.py`](scripts/evaluate_lora.py) and `merge_adapter: bool = False` to [`LoRADetector`](src/detectors/lora.py) (defaulting to `False` to preserve backwards compatibility).

![T4 latency merging speedup and prompt token budget](assets/04_t4_latency_merging.png)

---

## 3. External Out-of-Distribution (OOD) Generalization Study

Because near-perfect scores on one synthetic benchmark (`0.995–0.998` F1) do not prove cross-domain robustness, we ran a controlled 3-way evaluation (`0-shot v2` vs. `8-shot ICL s42` vs. `Merged LoRA`) on three external test suites using the exact same `Qwen/Qwen2.5-3B-Instruct` base and `ZERO_SHOT_V2` prompt contract on Tesla T4 (`float16`):

| Benchmark | Split / Composition | Method | Accuracy | Precision | Recall | F1 | Benign FPR | p50 Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **`agentic-prompt-injection-5k`** *(in-domain)* | Validation (`n=467`) | 0-shot v2 *(M1b)* | 0.8822 | 0.9534 | 0.8636 | 0.9063 | 0.0818 | — |
| | | 8-shot ICL `s42` *(M2)* | 0.9443 | 0.9608 | 0.9545 | 0.9577 | 0.0755 | — |
| | | **Merged LoRA (T4)** | **0.9936** | **0.9904** | **1.0000** | **0.9952** | **0.0189** | **0.388 s** |
| **`xTRam1/safe-guard-prompt-injection`** *(OOD #1)* | External test (`n=200`, 100/100) | 0-shot v2 | 0.9200 | 0.9200 | **0.9200** | 0.9200 | 0.0800 | **0.354 s** |
| | | **8-shot ICL `s42`** | **0.9397** | **0.9888** | 0.8889 | **0.9362** | **0.0100** | 0.545 s |
| | | Merged LoRA | 0.8543 | 0.8989 | 0.8000 | 0.8466 | 0.0909 | 0.382 s |
| **`jackhhao/jailbreak-classification`** *(OOD #2)* | External test (`n=189`, 89/100) | 0-shot v2 | 0.8519 | 0.7798 | 0.9551 | 0.8586 | 0.2400 | **0.401 s** |
| | | 8-shot ICL `s42` | 0.9202 | **0.9405** | 0.8876 | 0.9133 | **0.0505** | 0.594 s |
| | | **Merged LoRA** | **0.9312** | 0.8878 | **0.9775** | **0.9305** | 0.1100 | 0.425 s |
| **`agentic-ood-24`** *(OOD #3, human-written)* | External probes (`n=24`, 12/12) | 0-shot v2 | 0.7917 | 0.8889 | 0.6667 | 0.7619 | **0.0833** | **0.367 s** |
| | | **8-shot ICL `s42`** | **0.9167** | **0.9167** | 0.9167 | **0.9167** | **0.0833** | 0.566 s |
| | | Merged LoRA | 0.8333 | 0.7500 | **1.0000** | 0.8571 | 0.3333 | 0.387 s |

*Note: Across all 1,242 external OOD runs, 0–2 malformed outputs occurred per arm on unusual external strings and were excluded from metrics rather than coerced. On a fourth auxiliary benchmark (`deepset/prompt-injections`, `n=116`), Base 0-shot v2 scored `0.5977` F1 / `0.0179` FPR vs. Merged LoRA `0.5814` F1 / `0.0179` FPR.*

![In-domain vs. 3 external OOD benchmarks for 0-shot, 8-shot ICL, and Merged LoRA](assets/05_ood_generalization.png)

---

## 4. Root-Cause Error Analysis & Actionable Remedies

1. **In-domain specialization vs. OOD calibration trade-off.** LoRA did not fail uniformly OOD: it won F1 (`0.9305`) and Recall (`97.8%`) on `jailbreak-classification` and achieved **100% Recall (`12/12`)** on `agentic-ood-24` (catching buried needle-in-a-haystack resume/HTML-comment attacks and memory-poisoning prompts missed by 0-shot). However, frozen **8-shot ICL** preserved dramatically lower benign FPR on every external set (`1.0%`, `5.1%`, `8.3%`) because its 3.1B pretrained weights remained untouched.
2. **Synthetic shortcut learning.** Single-corpus LoRA drove validation loss down to `0.0027`, over-sharpening on synthetic phrasing patterns. In `agentic-ood-24`, LoRA flagged a legitimate human Slack approval (*"I signed off on your $450 monitor request in Workday"*) and a SQL cleanup migration (`DELETE FROM temp_sessions ... Reviewed in PR #412`), because short approvals and SQL snippets in `agentic-prompt-injection-5k` are overwhelmingly attacks.
3. **Label-taxonomy mismatch.** `agentic-prompt-injection-5k` explicitly labels `legitimate_roleplay_request` and `quoting_attack_for_analysis` as benign (`0`), whereas `safe-guard-prompt-injection` and `deepset/prompt-injections` label persona/roleplay prompts (e.g., *"You are CryptoMaster..."*, *"act as an interviewer"*) as attacks (`1`), causing `20` false negatives on `safe-guard`.

### Production Remedies (Milestones 4–5)
- **Multi-source training mixture + benign replay:** Blend agentic injections, direct jailbreaks, and diverse real-world benign emails, approvals, and code so LoRA retains ICL's cross-domain FPR calibration alongside LoRA's in-domain recall and merged speed.
- **Regularization:** Tune lower rank, fewer adapted modules, or earlier stopping against an external OOD dev slice.
- **Defense-in-depth:** Always pair a merged text classifier with task-aware context (user goal + provenance) and deterministic tool-authorization policy gates.
