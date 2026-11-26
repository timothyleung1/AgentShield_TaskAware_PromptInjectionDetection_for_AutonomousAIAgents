"""External OOD evaluation for AgentShield Milestone 3 (public Hugging Face benchmarks).

Compares three same-base (`Qwen/Qwen2.5-3B-Instruct`) detection arms:
  1. Zero-shot (`ZERO_SHOT_V2`)
  2. Frozen 8-shot ICL (`random_seed42_8shot` manifest from Milestone 2)
  3. Merged LoRA adapter (`PeftModel.merge_and_unload()`)

Evaluates strictly on public external out-of-distribution (OOD) benchmarks
without touching the sealed in-domain `agentic-prompt-injection-5k` test split:
  - `xTRam1/safe-guard-prompt-injection` (balanced test slice, n=200)
  - `jackhhao/jailbreak-classification` (balanced test slice, n=190)
  - Optional: `deepset/prompt-injections` (official test split, n=116)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from milestones.milestone3.src.detectors.lora import parse_label
from milestones.milestone3.src.training import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    EFFECTIVE_SYSTEM_PROMPT,
    build_messages,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "milestones"
    / "milestone2"
    / "manifests"
    / "random_seed42_8shot.json"
)
DEFAULT_ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "lora-qwen2.5-3b-t4"
    / "best_adapter"
)


def compute_metrics(records: list[dict]) -> dict:
    tp = sum(1 for r in records if r["actual"] == 1 and r["pred"] == 1)
    fp = sum(1 for r in records if r["actual"] == 0 and r["pred"] == 1)
    tn = sum(1 for r in records if r["actual"] == 0 and r["pred"] == 0)
    fn = sum(1 for r in records if r["actual"] == 1 and r["pred"] == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) else None
    accuracy = (tp + tn) / len(records) if records else 0.0
    lats = [r["latency"] for r in records if "latency" in r]
    return {
        "n": len(records),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4) if fpr is not None else None,
        "p50_latency_s": round(float(np.percentile(lats, 50)), 4) if lats else None,
    }


def build_few_shot_messages(text: str, demos: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": EFFECTIVE_SYSTEM_PROMPT}]
    for demo in demos:
        messages.append(
            {
                "role": "user",
                "content": f"Classify this untrusted text:\n\n{demo['text']}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps({"label": int(demo["label"])}),
            }
        )
    messages.append(
        {"role": "user", "content": f"Classify this untrusted text:\n\n{text}"}
    )
    return messages


def load_external_datasets(
    seed: int = 42, include_deepset: bool = False
) -> dict[str, list[dict]]:
    from datasets import load_dataset

    rng = random.Random(seed)
    suites: dict[str, list[dict]] = {}

    sg = list(load_dataset("xTRam1/safe-guard-prompt-injection", split="test"))
    sg_pos = [r for r in sg if int(r["label"]) == 1]
    sg_neg = [r for r in sg if int(r["label"]) == 0]
    rng.shuffle(sg_pos)
    rng.shuffle(sg_neg)
    suites["safe-guard-prompt-injection"] = [
        {"id": f"sg_{i}", "text": r["text"], "label": int(r["label"])}
        for i, r in enumerate(sg_pos[:100] + sg_neg[:100])
    ]

    jb = list(load_dataset("jackhhao/jailbreak-classification", split="test"))
    jb_pos = [r for r in jb if r["type"] == "jailbreak"]
    jb_neg = [r for r in jb if r["type"] == "benign"]
    rng.shuffle(jb_pos)
    rng.shuffle(jb_neg)
    suites["jailbreak-classification"] = [
        {
            "id": f"jb_{i}",
            "text": r["prompt"],
            "label": 1 if r["type"] == "jailbreak" else 0,
        }
        for i, r in enumerate(jb_pos[:100] + jb_neg[:100])
    ]

    if include_deepset:
        ds = list(load_dataset("deepset/prompt-injections", split="test"))
        suites["deepset-prompt-injections"] = [
            {"id": f"ds_{i}", "text": r["text"], "label": int(r["label"])}
            for i, r in enumerate(ds)
        ]

    return suites


def evaluate_arm(
    model,
    tokenizer,
    device: str,
    rows: list[dict],
    demos: list[dict] | None = None,
    max_new_tokens: int = 24,
) -> tuple[dict, int]:
    import torch

    records: list[dict] = []
    errors = 0
    for row in rows:
        messages = (
            build_few_shot_messages(row["text"], demos)
            if demos
            else build_messages(row["text"])
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3584
        ).to(device)
        n_in = int(inputs["input_ids"].shape[-1])
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        lat = time.perf_counter() - t0
        raw = tokenizer.decode(out[0, n_in:], skip_special_tokens=True).strip()
        try:
            pred = parse_label(raw)
        except ValueError:
            errors += 1
            continue
        records.append(
            {
                "id": row["id"],
                "actual": int(row["label"]),
                "pred": pred,
                "latency": lat,
            }
        )
    metrics = compute_metrics(records)
    metrics["format_errors"] = errors
    return metrics, errors


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(
        description="Run 3-way external OOD benchmark: 0-shot v2 vs. 8-shot ICL vs. Merged LoRA."
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--include-deepset", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    dtype = (
        torch.float16
        if device == "cuda"
        else (torch.bfloat16 if device == "mps" else torch.float32)
    )

    suites = load_external_datasets(
        seed=args.seed, include_deepset=args.include_deepset
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    demos = manifest["examples"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=BASE_MODEL_REVISION, torch_dtype=dtype
        )
        .to(device)
        .eval()
    )

    results: dict[str, dict[str, dict]] = {}
    for arm_name, arm_demos in (("0-shot v2", None), ("8-shot ICL s42", demos)):
        results[arm_name] = {}
        for suite_name, rows in suites.items():
            metrics, _ = evaluate_arm(base, tokenizer, device, rows, demos=arm_demos)
            results[arm_name][suite_name] = metrics
            print(f"{suite_name:30s} | {arm_name:15s} | {json.dumps(metrics)}")

    merged = (
        PeftModel.from_pretrained(base, args.adapter)
        .merge_and_unload()
        .to(device)
        .eval()
    )
    results["Merged LoRA"] = {}
    for suite_name, rows in suites.items():
        metrics, _ = evaluate_arm(merged, tokenizer, device, rows, demos=None)
        results["Merged LoRA"][suite_name] = metrics
        print(f"{suite_name:30s} | {'Merged LoRA':15s} | {json.dumps(metrics)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
