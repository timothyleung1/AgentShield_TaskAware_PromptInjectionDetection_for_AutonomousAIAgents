# AgentShield — Presentation Slides & Speaker Notes (CMU 11-967 HW12)

- **Google Slides Deck:** https://docs.google.com/presentation/d/1bJqjEwW5HzluILUPbl5a4iE3-9sSiQyaApkkNDlxnsg/edit
- **Exported Slide Copies:** [`AgentShield_Presentation.pdf`](AgentShield_Presentation.pdf) | [`AgentShield_Presentation.pptx`](AgentShield_Presentation.pptx)
- **Authors:** Eva Liu & Timothy Leung
- **Rubric Coverage:**
  - **(a) The Task:** Slide 2 (`Prompt-Injection Classifier Goal`)
  - **(b) Data:** Slide 3 (`Dataset Details & Splits`)
  - **(c) Experimental Design:** Slide 4 (`Experimental Design & Methods`)
  - **(d) Preliminary Insights:** Slides 5–10 (Dev ICL scaling, Locked Test headline & family breakdown, LoRA weight-merging latency speedup, 3-benchmark OOD reality check, Consolidated insights) + Slide 11 (`Conclusions & Next Steps`)

---

## Slide-by-Slide Outline & Speaker Notes

### Slide 1 — AgentShield (Cover)
- **Headline:** In-Context Learning vs. LoRA Fine-Tuning for Prompt-Injection Detection in Autonomous AI Agents (`Qwen/Qwen2.5-3B-Instruct` · `agentic-prompt-injection-5k`).
- **Speaker Notes:** Welcome! Today we present AgentShield: a controlled empirical comparison between in-context learning (ICL) and LoRA fine-tuning on the same 3B open-weight model (`Qwen2.5-3B-Instruct`) for detecting prompt injections in untrusted agent-visible text. We cover the task, dataset, experimental design, sealed test results, a latency engineering discovery on LoRA weight merging, and an out-of-distribution generalization check across three external benchmarks.

### Slide 2 — `(a) THE TASK`: Prompt-Injection Classifier Goal
- **Visual:** 4-stage pipeline (`Untrusted Text → AgentShield → Binary Verdict {"label": 0|1} → Policy Gate (M4)`).
- **Key Points:** Binary verdict (`0 = benign`, `1 = injection`); catches overrides, authority spoofing, and memory poisoning while allowing benign security quotes; strictly text-only classification (not live tool execution or end-to-end authorization).
- **Speaker Notes:** Autonomous agents ingest untrusted text from documents, emails, and tool outputs. Our core task is text-only binary classification: does untrusted text attempt to hijack instructions or spoof authority? We keep the scope honest — the benchmark provides only text and a binary label, not a separate user goal or tool runtime, and in production a classifier must feed into an independent deterministic policy gate.

### Slide 3 — `(b) DATA`: Dataset Details & Splits
- **Key Metrics:** `3,923` Train (`55.1%` attack) · `467` Validation (`66.0%` attack) · `545` Sealed Test (`51.4%` attack) from `3nesdeniz/agentic-prompt-injection-5k` (pinned rev `37ecd4e7`).
- **Audit & OOD Data:** 4 attack families (`Approval`, `RAG`, `Memory`, `Tool`), `0` exact duplicates across splits, `0` overlapping techniques between train and val. External OOD stress sets: `safe-guard-prompt-injection` (`n=200`), `jailbreak-classification` (`n=189`), and `agentic-ood-24` (`24` human-written probes).
- **Speaker Notes:** We retained the provider's official splits (`3,923` train, `467` validation, `545` sealed test). Our leakage audit verified zero exact duplicates and zero overlapping attack techniques between train and validation across all four families. To evaluate generalization beyond a single synthetic corpus, we added three external out-of-distribution test sets.

### Slide 4 — `(c) EXPERIMENTAL DESIGN`: Experimental Design & Methods
- **Controlled Base:** `Qwen/Qwen2.5-3B-Instruct` (rev `aa8e7253`) · `ZERO_SHOT_V2` prompt · Greedy JSON `{"label": 0|1}` (`max_new_tokens=24`).
- **3 Adaptation Arms:**
  1. **Zero-Shot v2:** Frozen base weights, prompt rules only (`~346` input tokens).
  2. **Few-Shot ICL:** Frozen base weights + 4/8 train demonstrations (`~829` tokens = `2.4×`; dev winner `random_seed42_8shot`).
  3. **LoRA Fine-Tuning:** `r=8`, `α=16`, 7 linear layers, masked completion loss (`~346` tokens, merged at load; dev winner Epoch 1).
- **Evaluation Rigor:** Select on validation (`467`) → evaluate once on sealed test (`545`) with exact two-sided McNemar + 10,000 paired bootstrap CIs, plus T4 latency merging benchmarks and 3 external OOD sets.

### Slide 5 — `(d) INSIGHTS · DEV`: Dev: Prompting & ICL Scaling
- **Chart:** Validation F1 & Benign FPR across `0-shot (v2)` (`90.6%` F1 / `8.2%` FPR), `4-shot balanced` (`91.1%` / `10.1%`), `8-shot balanced` (`87.0%` / `6.3%`), and `8-shot random s42` (`95.8%` / `7.5%`).
- **Takeaway:** `v1 → v2` prompt lifted F1 `+7.5 pp` and halved FPR; demonstration choice at `k=8` swung F1 by `8.7 pp`; `8-shot s42` neared `gpt-4o-mini` (`96.3%` F1) but still misclassified `11/12` benign quoted attacks.

### Slide 6 — `(d) PRELIMINARY INSIGHTS · LOCKED TEST`: LoRA Dominated Locked Test (`n = 545`)
- **Hero Metrics:** `0.998` LoRA Attack F1 (`+7.96 pp` vs. `0.919` ICL, 95% CI `[+5.8, +10.4]`), `0.38%` Benign FPR (`1/265` vs. `39/265` ICL, `−14.34 pp`), `48 / 1` fixed vs. regressed (`p = 1.78e-13`).

### Slide 7 — `(d) INSIGHTS · TEST`: Locked Test: Metrics & Families
- **Chart:** All 5 locked-test metrics with 95% paired bootstrap CIs + 4 subgroup summary cards (`Approval`: `90.5% → 100%`, `RAG`: `98.8% → 100%`, `Memory & Tool`: both `100%`, `Benign`: `39 FP → 1 FP`).

### Slide 8 — `(d) INSIGHTS · SPEED`: Merging LoRA Cuts Latency 1.92×
- **Chart:** [`../milestones/milestone3/assets/04_t4_latency_merging.png`](../milestones/milestone3/assets/04_t4_latency_merging.png)
- **Key Finding:** Independent GCP Tesla T4 reproduction (`fp16`, 1 epoch, 41 min, val F1 `0.995`, 100% recall). Unmerged `PeftModel` wrapped 252 linear layers (`0.78s` p50); `.merge_and_unload()` cut latency `1.92×` to `0.40s` p50 with `30/30` exact agreement — ~30% faster than 8-shot ICL (`0.55s`) with 58% fewer prompt tokens.

### Slide 9 — `(d) INSIGHTS · OOD`: OOD Check: 3 External Benchmarks
- **Chart:** [`../milestones/milestone3/assets/05_ood_generalization.png`](../milestones/milestone3/assets/05_ood_generalization.png)
- **Key Metrics:**
  - `safe-guard` (`n=200`): 8-shot ICL won (`93.6%` F1, `1.0%` FPR); LoRA dropped to `84.7%` F1.
  - `jailbreak` (`n=189`): Merged LoRA won F1 (`93.1%`) & Recall (`97.8%`); ICL kept half the FPR (`5.1%`).
  - `agentic-24` (`n=24`): LoRA hit `100%` Recall (haystacks & memory poisoning); ICL won F1 (`91.7%`, `8.3%` FPR).

### Slide 10 — `(d) INSIGHTS`: Key Insights: ICL vs. LoRA
- **What the Results Show:** In-domain LoRA wins quality (`+8 pp` F1) and speed (~30% merged); single-dataset LoRA overfits synthetic cues OOD; frozen ICL preserves `1–8%` FPR across all new domains; label taxonomies differ across public benchmarks.
- **How to Improve Next:** Train LoRA on a multi-source mix + real benign replay, regularize to prevent shortcuts, always merge adapter weights for serving, and pair the detector with task context & deterministic policy gates.

### Slide 11 — Conclusions & Next Steps
- **Summary:** (1) In-domain winner (`0.998` F1, `0.4%` FPR, ~30% faster merged), (2) OOD reality (frozen ICL kept `1–8%` OOD FPR), (3) M4–M5 roadmap (multi-source LoRA + task context + deterministic policy gates).
