# AgentShield — Presentation Slides & Speaker Notes (CMU 11-967 HW12)

- **Google Slides Deck:** https://docs.google.com/presentation/d/1tI4yEuXuEQpULPIdgxdZRGQ5c4g2fO-GEtX4uuaRPas/edit
- **Exported Slide Copies:** [`AgentShield_Presentation.pdf`](AgentShield_Presentation.pdf) | [`AgentShield_Presentation.pptx`](AgentShield_Presentation.pptx)
- **Authors:** Eva Liu & Timothy Leung
- **Rubric Coverage:**
  - **(a) The Task:** Slide 1 (`AgentShield` cover + pipeline + scope)
  - **(b) Data:** Slide 2 (`Our Dataset: agentic-5k`) & Slide 3 (`Other PI Datasets & Differences — Eval Only`)
  - **(c) Experimental Design:** Slide 4 (`Baseline & In-Context Learning`), Slide 5 (`LoRA: Fast, Lightweight Fine-Tuning`), & Slide 6 (`LoRA Rank Scaling: Size & Time`)
  - **(d) Preliminary Insights:** Slides 7–9 (`LoRA Results: Quality & Speed`, `Trained LoRA on Unseen Datasets`, `Key Insights & Takeaways`) + Slide 10 (`Summary & Questions`)

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

### Slide 5 — LoRA: Fast, Lightweight Fine-Tuning `(c) EXPERIMENTAL DESIGN`
- **Visual Diagram:** `Frozen Base Model (W, 3.09B)` + `Trainable LoRA (A × B, 15M params = 0.49%)` → `Merged Weights (W' = W + ΔW, zero overhead)`.
- **4 Efficiency Hero Stats:** **`41 min`** (1 full epoch over `3,923` rows on a single budget Tesla T4) · **`0.49%`** trainable params (`14.97M` of `3.09B`) · **`8.2 GB`** peak GPU VRAM (fits on 1× 16 GB T4) · **`57 MB`** saved adapter file (>100× smaller than a 6.2 GB full checkpoint).
- **Controlled Setup & Convergence:** Same `Qwen2.5-3B` base & `v2` prompt (0 demos), `r=8`, `α=16` on 7 linear layers, masked loss on assistant JSON label only; converged in 1 epoch (`491` steps, lowest val loss), verified on both T4 (`fp16`) and A10G (`bf16`).
- **Speaker Notes:** The whole point of LoRA is fast, cheap, lightweight adaptation. Instead of full fine-tuning across 3.09 billion parameters, we froze 99.5% of Qwen2.5-3B and trained just 15 million rank-8 adapter parameters (`0.49%`). On a single entry-level NVIDIA Tesla T4 GPU, one complete epoch over all `3,923` training rows finished in just `41` minutes using only `8.2 GB` of peak VRAM — and Epoch 1 was already our best validation checkpoint! Even better, the saved adapter is only `57 MB` (>100× smaller than a full `6.2 GB` checkpoint) and folds back into the base weights for zero extra inference overhead.

### Slide 6 — LoRA Rank Scaling: Size & Time `(c) EXPERIMENTAL DESIGN`
- **Scaling Table (`Qwen/Qwen2.5-3B-Instruct`, 7 linear modules, anchored to measured Tesla T4 `r=8`):**
  - `LoRA r = 4`: `7.5M (0.24%)` · `29 MB` adapter · `~86 MB` train ckpt · `~8.1 GB` VRAM · `~40.7 min` *(est.)*
  - **`LoRA r = 8` (Ours):** `15.0M (0.49%)` · **`57 MB`** adapter · `~171 MB` train ckpt · **`8.2 GB`** VRAM · **`40.9 min`** *(actual)*
  - `LoRA r = 16`: `29.9M (0.97%)` · `114 MB` adapter · `~343 MB` train ckpt · `~8.4 GB` VRAM · `~41.4 min` *(est.)*
  - `LoRA r = 32`: `59.9M (1.94%)` · `228 MB` adapter · `~685 MB` train ckpt · `~8.9 GB` VRAM · `~42.4 min` *(est.)*
  - `Full Fine-Tuning`: `3.09B (100%)` · `~6.2 GB (fp16)` · `~35 GB` train ckpt · `~39–51 GB` VRAM · **OOM on 16 GB T4**
- **Speaker Notes:** A common question is how to choose LoRA rank `r` and how training time and checkpoint size scale. Anchored to our measured Tesla T4 run at `r=8` (`40.9` minutes, `8.2 GB` peak VRAM, `57 MB` `fp32` adapter, `~171 MB` training checkpoint with AdamW states), adapter and checkpoint sizes scale linearly with `r` (`~7.1 MB` per rank). Surprisingly, 1-epoch training time stays nearly flat across `r=4` to `32` (`~41` to `42.5` minutes) because `~96%` of each step's compute is propagating activations through the frozen `3.09B` base model. By contrast, full fine-tuning needs `~39–51 GB` VRAM (OOMing a single 16 GB T4) and `~35 GB` checkpoints. Practically: if LoRA underfits in-domain, raising `r` is almost free in time; if it overfits OOD, keep `r` small and diversify data.

### Slide 7 — LoRA Results: Quality & Speed `(d) PRELIMINARY INSIGHTS`
- **Sealed Test (`n=545`, evaluated once):** `99.8%` Attack F1 (vs. `91.9%` ICL), `0.4%` Benign False Alarms (`1` FP vs. `39` FPs), `0` Missed Attacks (vs. `9` missed; `p = 1.78e-13`).
- **Speed Finding:** Merging LoRA weights (`.merge_and_unload()`) cut Tesla T4 latency `1.92×` (`0.78s → 0.40s`) — ~30% faster than 8-shot ICL (`0.55s`) with 58% fewer prompt tokens and `30/30` identical outputs.
- **Speaker Notes:** On the 545-row held-out test set — evaluated once after both models were frozen — LoRA decisively beat 8-shot ICL: `99.8%` F1 versus `91.9%`, cutting false alarms from `39` down to `1` and catching all `280` attacks (`48` ICL errors fixed, `1` regression, McNemar `p = 1.78e-13`). We also found a practical engineering win: leaving PEFT adapters unmerged made LoRA slower than ICL, whereas calling `merge_and_unload()` cut T4 latency `1.92×` (`0.78s` down to `0.40s`) with identical predictions — ~30% faster than 8-shot ICL using 58% fewer prompt tokens.

### Slide 8 — Trained LoRA on Unseen Datasets `(d) INSIGHTS · OOD`
- **Chart:** [`../milestones/milestone3/assets/05_ood_generalization.png`](../milestones/milestone3/assets/05_ood_generalization.png)
- **Public External Results:**
  - `safe-guard` (`n=200`): 8-shot ICL won (`94%` F1, `1%` FPR); LoRA dropped to `85%` F1.
  - `jailbreak` (`n=189`): LoRA led F1 (`93%`) & Recall (`98%`); ICL kept half the FPR (`5%`).
  - `deepset` (`n=116`): LoRA flat vs. 0-shot (`58%` vs. `60%` F1) due to roleplay label definition clash.
- **Speaker Notes:** Next, we tested our exact same trained LoRA model on the external Hugging Face datasets we never trained on. The contrast is eye-opening: on `jailbreak-classification`, LoRA still achieved the highest F1 (`93%`) and recall (`98%`), though 8-shot ICL had half the false alarms (`5%` vs `11%`). On `safe-guard`, frozen 8-shot ICL beat LoRA by 9 F1 points (`94%` vs `85%`) with only `1%` false positives. And on `deepset`, both 0-shot and LoRA stayed around `58–60%` F1 because `deepset` labels *"act as an interviewer"* prompts as attacks.

### Slide 9 — Key Insights & Takeaways `(d) INSIGHTS`
- **What We Learned:** In-domain LoRA dominates quality & speed; out-of-domain frozen ICL keeps lower FPR (`1–5%`); OOD shift came from single-corpus shortcuts & differing roleplay/quote label rules; always merge LoRA weights for serving.
- **What We Would Do Next:** Train LoRA on a multi-source mix + real benign text, regularize (lower rank / replay), add task context (user goal + provenance), and pair any detector with deterministic policy gates.

### Slide 10 — Summary & Questions
- **Wrap-Up:** (1) Dataset & Design (`agentic-5k` only, `Qwen2.5-3B`), (2) LoRA vs. ICL (`99.8%` F1 & ~30% faster merged after ~41 min training), (3) Big Takeaway (frozen ICL kept `1–5%` OOD FPR → multi-source LoRA + policy gates next).
