import argparse
from pathlib import Path
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from milestones.milestone3.src.detectors.lora import parse_label
from milestones.milestone3.src.training import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    build_messages,
)

DEFAULT_ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "lora-qwen2.5-3b-t4"
    / "best_adapter"
)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive local CLI for testing a merged Qwen2.5-3B LoRA detector."
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--text", default=None, help="Classify a single text and exit.")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "mps" else (torch.float16 if device == "cuda" else torch.float32)

    print(f"Loading {BASE_MODEL} + merged LoRA adapter ({args.adapter}) on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_MODEL_REVISION, dtype=dtype
    )
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    model.to(device).eval()
    if model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    def classify(text: str):
        prompt = tokenizer.apply_chat_template(
            build_messages(text), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        n_in = int(inputs["input_ids"].shape[-1])
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        lat = time.perf_counter() - t0
        raw = tokenizer.decode(out[0, n_in:], skip_special_tokens=True).strip()
        label = parse_label(raw)
        verdict = "INJECTION (1)" if label == 1 else "BENIGN (0)"
        print(f"Verdict: {verdict} | raw={raw} | tokens={n_in} | latency={lat:.3f}s\n")

    if args.text is not None:
        classify(args.text)
        return

    print("Ready! Paste untrusted text and press Enter twice to classify (Ctrl-C to exit).\n")
    while True:
        try:
            print(">>> Enter text (blank line to submit):")
            lines = []
            while True:
                line = input()
                if line == "" and lines:
                    break
                lines.append(line)
            text = "\n".join(lines).strip()
            if text:
                classify(text)
        except (EOFError, KeyboardInterrupt):
            print("\nDone.")
            break


if __name__ == "__main__":
    main()
