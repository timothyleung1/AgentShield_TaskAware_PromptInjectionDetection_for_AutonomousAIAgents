# AgentShield — Presentation Slides & Speaker Notes (CMU 11-967 HW12)

- **Google Slides Deck:** https://docs.google.com/presentation/d/1tI4yEuXuEQpULPIdgxdZRGQ5c4g2fO-GEtX4uuaRPas/edit
- **Exported Slide Copies:** [`AgentShield_Presentation.pdf`](AgentShield_Presentation.pdf) | [`AgentShield_Presentation.pptx`](AgentShield_Presentation.pptx)
- **Authors:** Eva Liu & Timothy Leung
- **Rubric Coverage:**
  - **(a) The Task:** Slide 1 (`AgentShield` cover + pipeline + scope)
  - **(b) Data:** Slide 2 (`Our Dataset: agentic-5k`) & Slide 3 (`Other PI Datasets & Differences — Eval Only`)
  - **(c) Experimental Design:** Slide 4 (`Baseline & In-Context Learning`) & Slide 5 (`LoRA: How It Works & Setup`)
  - **(d) Preliminary Insights:** Slides 6–8 (`LoRA Results: Quality & Speed`, `Trained LoRA on Unseen Datasets`, `Key Insights & Takeaways`) + Slide 9 (`Summary & Questions`)

---

## Slide-by-Slide Outline & Speaker Notes

### Slide 1 — AgentShield `(a) THE TASK`
- **Visual:** 4-stage pipeline (`Untrusted Text → AgentShield (Qwen2.5-3B) → Binary Verdict {"label": 0|1} → Policy Gate`) with explicit scope note (text-only binary classification, not live tool authorization).
- **Speaker Notes:** Welcome! Today we present AgentShield. Autonomous AI agents read untrusted text from documents, emails, and tool outputs. Our goal is a text-only binary classifier on `Qwen2.5-3B-Instruct` that flags prompt-injection attempts (`1`) while allowing benign text (`0`). We keep scope honest: this classifier feeds into an independent policy gate and is not end-to-end tool authorization by itself. Let's start with the dataset we chose.

### Slide 2 — Our Dataset: `agentic-5k` `(b) DATA`
- **Splits:** `3,923` Train (`55%` attack) · `467` Validation (`66%` attack) · `545` Sealed Test (`51%` attack) from `3nesdeniz/agentic-prompt-injection-5k`.
- **What's Inside:** 4 agentic attack families (*Approval/Authority*, *RAG Document*, *Memory Poisoning*, *Tool Abuse*) + hard benign examples (quoted attacks & safe roleplay = `0`); short synthetic enterprise text (~190 chars); 0 duplicates & 0 overlapping attack techniques between train and val.
- **Speaker Notes:** We chose `3nesdeniz/agentic-prompt-injection-5k` on Hugging Face (`4,935` total examples across official train, validation, and sealed test splits). Unlike simple chatbot jailbreak datasets, it targets four realistic agentic attack families: fake approvals, instructions hidden in RAG documents, memory poisoning, and tool abuse. Its benign class intentionally includes hard negatives like security discussions that quote an attack and safe roleplay requests. Our audit confirmed zero duplicate texts and zero overlapping attack techniques between train and validation.

### Slide 3 — Other PI Datasets & Differences `(b) DATA`
- **Training Boundary:** Trained **only** on `agentic-prompt-injection-5k`; `xTRam1/safe-guard-prompt-injection` (`n=200`), `jackhhao/jailbreak-classification` (`n=189`), and `deepset/prompt-injections` (`n=116`) were **never** used for training (evaluation-only).
- **Key Contrast:** Those external benchmarks target direct chatbot overrides/jailbreaks and often label persona or *"Act as..."* prompts as attacks (`1`), whereas `agentic-5k` targets indirect enterprise agent attacks and labels roleplay/quoted attacks as benign (`0`).
- **Speaker Notes:** Before looking at models, it helps to compare our chosen dataset with other popular public prompt-injection benchmarks on Hugging Face: `safe-guard-prompt-injection`, `jailbreak-classification`, and `deepset/prompt-injections`. Those focus on direct chatbot jailbreaks and often label persona or *"act as"* prompts as attacks, whereas `agentic-5k` focuses on indirect enterprise agent attacks and treats roleplay and quoted attacks as benign. Crucially, we did NOT train on any of those other datasets — we used them strictly at the end to test out-of-distribution generalization.

### Slide 4 — Baseline & In-Context Learning `(c) DESIGN · (d) INSIGHTS`
- **Baseline:** Frozen `Qwen/Qwen2.5-3B-Instruct` + trust-boundary prompt (`v2`) + strict JSON output.
- **ICL Results (Validation `n=467`):** Demo choice swung F1 from `87% → 96%` (`8-shot balanced` vs. `8-shot random s42`, matching `gpt-4o-mini`), at a cost of `2.4×` prompt tokens and `11/12` quoted-attack false positives.
- **Speaker Notes:** We used frozen `Qwen/Qwen2.5-3B-Instruct` as our baseline with our trust-boundary v2 prompt and strict JSON output. Without changing any weights, we tested adding 4 or 8 labeled training examples directly into the prompt (in-context learning). The biggest surprise was demonstration sensitivity: two valid 8-shot sets swung validation F1 by nearly 9 points, from `87%` to `96%` — matching `gpt-4o-mini`. However, 8-shot ICL more than doubled prompt length (`2.4×` tokens) and still misclassified `11` of `12` benign security quotes.

### Slide 5 — LoRA: How It Works & Setup `(c) EXPERIMENTAL DESIGN`
- **Visual Diagram:** `Frozen Base Model (W)` + `Trainable LoRA (A × B, r=8, <1% params)` → `Merged Weights (W' = W + ΔW)`.
- **Setup & Training Time:** Same `Qwen2.5-3B` base & `v2` prompt (0 demos), `r=8`, `α=16` on 7 attention & MLP layers, masked loss on assistant JSON label only, trained on `3,923` train rows — **~41 minutes** for 1 full epoch on Tesla T4 (`fp16`, ~8.2 GB peak VRAM, 491 steps; also verified on A10G `bf16`, selecting Epoch 1 by lowest val loss).
- **Speaker Notes:** Instead of updating all 3.1 billion parameters, LoRA freezes the base Qwen weights and trains tiny rank-8 adapter matrices (<1% of parameters) on the attention and MLP projections. At deployment time, those matrices fold right back into the base weights for zero extra inference cost. We kept the exact same v2 prompt with zero in-context demos and computed loss only on the JSON label. Training 1 full epoch over all `3,923` rows took ~41 minutes on a single NVIDIA Tesla T4 using ~8.2 GB of VRAM, and we selected Epoch 1 by lowest validation loss.

### Slide 6 — LoRA Results: Quality & Speed `(d) PRELIMINARY INSIGHTS`
- **Sealed Test (`n=545`, evaluated once):** `99.8%` Attack F1 (vs. `91.9%` ICL), `0.4%` Benign False Alarms (`1` FP vs. `39` FPs), `0` Missed Attacks (vs. `9` missed; `p = 1.78e-13`).
- **Speed Finding:** Merging LoRA weights (`.merge_and_unload()`) cut Tesla T4 latency `1.92×` (`0.78s → 0.40s`) — ~30% faster than 8-shot ICL (`0.55s`) with 58% fewer prompt tokens and `30/30` identical outputs.
- **Speaker Notes:** On the 545-row held-out test set — evaluated once after both models were frozen — LoRA decisively beat 8-shot ICL: `99.8%` F1 versus `91.9%`, cutting false alarms from `39` down to `1` and catching all `280` attacks (`48` ICL errors fixed, `1` regression, McNemar `p = 1.78e-13`). We also found a practical engineering win: leaving PEFT adapters unmerged made LoRA slower than ICL, whereas calling `merge_and_unload()` cut T4 latency `1.92×` (`0.78s` down to `0.40s`) with identical predictions — ~30% faster than 8-shot ICL using 58% fewer prompt tokens.

### Slide 7 — Trained LoRA on Unseen Datasets `(d) INSIGHTS · OOD`
- **Chart:** [`../milestones/milestone3/assets/05_ood_generalization.png`](../milestones/milestone3/assets/05_ood_generalization.png)
- **Public External Results:**
  - `safe-guard` (`n=200`): 8-shot ICL won (`94%` F1, `1%` FPR); LoRA dropped to `85%` F1.
  - `jailbreak` (`n=189`): LoRA led F1 (`93%`) & Recall (`98%`); ICL kept half the FPR (`5%`).
  - `deepset` (`n=116`): LoRA flat vs. 0-shot (`58%` vs. `60%` F1) due to roleplay label definition clash.
- **Speaker Notes:** Next, we tested our exact same trained LoRA model on the external Hugging Face datasets we never trained on. The contrast is eye-opening: on `jailbreak-classification`, LoRA still achieved the highest F1 (`93%`) and recall (`98%`), though 8-shot ICL had half the false alarms (`5%` vs `11%`). On `safe-guard`, frozen 8-shot ICL beat LoRA by 9 F1 points (`94%` vs `85%`) with only `1%` false positives. And on `deepset`, both 0-shot and LoRA stayed around `58–60%` F1 because `deepset` labels *"act as an interviewer"* prompts as attacks.

### Slide 8 — Key Insights & Takeaways `(d) INSIGHTS`
- **What We Learned:** In-domain LoRA dominates quality & speed; out-of-domain frozen ICL keeps lower FPR (`1–5%`); OOD shift came from single-corpus shortcuts & differing roleplay/quote label rules; always merge LoRA weights for serving.
- **What We Would Do Next:** Train LoRA on a multi-source mix + real benign text, regularize (lower rank / replay), add task context (user goal + provenance), and pair any detector with deterministic policy gates.

### Slide 9 — Summary & Questions
- **Wrap-Up:** (1) Dataset & Design (`agentic-5k` only, `Qwen2.5-3B`), (2) LoRA vs. ICL (`99.8%` F1 & ~30% faster merged after ~41 min training), (3) Big Takeaway (frozen ICL kept `1–5%` OOD FPR → multi-source LoRA + policy gates next).
