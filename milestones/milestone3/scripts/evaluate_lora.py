import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from milestones.milestone2.src.icl.dataset import load_benchmark_split
from milestones.milestone3.src.detectors.lora import LoRADetector
from milestones.milestone3.src.training import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    DATASET_NAME,
    DATASET_REVISION,
    EFFECTIVE_SYSTEM_PROMPT,
    split_id_sha256,
    validate_split,
)

MILESTONE_ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = [
    "id",
    "actual",
    "predicted",
    "status",
    "latency_seconds",
    "attack_family",
    "technique",
    "input_tokens",
    "output_tokens",
    "error",
]


def calculate_metrics(rows):
    tp = fp = tn = fn = 0
    for row in rows:
        actual = int(row["actual"])
        predicted = int(row["predicted"])
        if actual == predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == predicted == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "samples": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(rows) if rows else None,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "fpr": fp / (fp + tn) if fp + tn else None,
    }


def build_summary(rows):
    successful = [row for row in rows if row["status"] == "success"]
    by_family = defaultdict(list)
    for row in successful:
        by_family[row["attack_family"] or "unknown"].append(row)

    latencies = [float(row["latency_seconds"]) for row in successful]
    input_tokens = [int(row["input_tokens"]) for row in successful]
    output_tokens = [int(row["output_tokens"]) for row in successful]
    return {
        "metrics": calculate_metrics(successful),
        "attack_family_metrics": {
            family: calculate_metrics(family_rows)
            for family, family_rows in sorted(by_family.items())
        },
        "operational": {
            "latency_seconds": {
                "mean": float(np.mean(latencies)) if latencies else None,
                "p50": float(np.percentile(latencies, 50)) if latencies else None,
                "p95": float(np.percentile(latencies, 95)) if latencies else None,
            },
            "tokens": {
                "mean_input": (
                    sum(input_tokens) / len(input_tokens)
                    if input_tokens
                    else None
                ),
                "mean_output": (
                    sum(output_tokens) / len(output_tokens)
                    if output_tokens
                    else None
                ),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA adapter on validation or final test."
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to a trained adapter directory.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--merge-adapter",
        action="store_true",
        help="Fold LoRA weights into the base model via merge_and_unload() for faster inference.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Use test only for the final frozen evaluation.",
    )
    args = parser.parse_args()

    if not args.adapter.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter}")
    if args.split == "test" and args.max_new_tokens != 24:
        raise ValueError("--max-new-tokens cannot change for the frozen test run.")

    validation = list(
        load_benchmark_split(args.split, allow_test=(args.split == "test"))
    )
    validate_split(validation, args.split)
    if args.split == "test" and len(validation) != 545:
        raise ValueError(
            f"The final test run must contain exactly 545 rows, got {len(validation)}."
        )
    output_path = (
        MILESTONE_ROOT
        / "data"
        / "reports"
        / f"lora_{args.split}_{args.adapter.name}.csv"
    )
    metadata_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": args.split,
        "sample_count": len(validation),
        "sample_ids_sha256": split_id_sha256(validation),
        "test_split_accessed": args.split == "test",
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_MODEL_REVISION,
        "adapter": str(args.adapter.resolve()),
        "merge_adapter": args.merge_adapter,
        "max_new_tokens": args.max_new_tokens,
        "requested_device": args.device,
        "requested_dtype": args.dtype,
        "prompt_version": "ZERO_SHOT_V2",
        "prompt_sha256": hashlib.sha256(
            EFFECTIVE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
    }
    rows = []
    started_at = datetime.now(timezone.utc).isoformat()
    if output_path.exists() or metadata_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output already exists; pass --resume to continue: {output_path}"
            )
        if not output_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("Both checkpoint CSV and metadata JSON are required.")
        old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in config.items():
            if old_metadata.get(key) != value:
                raise RuntimeError(f"Cannot resume: {key} does not match.")
        started_at = old_metadata["started_at"]
        with output_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != CSV_FIELDS:
                raise RuntimeError(
                    f"Checkpoint has unexpected columns: {reader.fieldnames}"
                )
            rows = list(reader)

    completed = {
        str(row["id"]): row for row in rows if row["status"] == "success"
    }
    rows = list(completed.values())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    detector = LoRADetector(
        args.adapter,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        merge_adapter=args.merge_adapter,
    )

    for index, example in enumerate(validation):
        example_id = str(example["id"])
        if example_id in completed:
            continue
        started = time.perf_counter()
        try:
            result = detector.predict(example["text"])
            row = {
                "id": example_id,
                "actual": int(example["label"]),
                "predicted": result.label,
                "status": "success",
                "latency_seconds": round(result.latency_seconds, 4),
                "attack_family": example.get("attack_family", ""),
                "technique": example.get("technique", ""),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "error": "",
            }
        except Exception as exc:
            row = {
                "id": example_id,
                "actual": int(example["label"]),
                "predicted": "",
                "status": type(exc).__name__,
                "latency_seconds": round(time.perf_counter() - started, 4),
                "attack_family": example.get("attack_family", ""),
                "technique": example.get("technique", ""),
                "input_tokens": "",
                "output_tokens": "",
                "error": str(exc),
            }
            print(f"Failed id={example_id}: {exc}", file=sys.stderr)

        rows.append(row)
        with output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

        summary = build_summary(rows)
        metadata = {
            **config,
            "runtime": {
                "device": detector.device,
                "dtype": detector.dtype,
                "python_version": sys.version,
                "transformers_version": detector.transformers_version,
                "peft_version": importlib.metadata.version("peft"),
            },
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "completed_count": sum(r["status"] == "success" for r in rows),
            "failure_count": sum(r["status"] != "success" for r in rows),
            **summary,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Sample {index + 1}/{len(validation)} id={example_id} "
            f"status={row['status']}"
        )

    successful = [row for row in rows if row["status"] == "success"]
    summary = build_summary(successful)
    metadata = {
        **config,
        "runtime": {
            "device": detector.device,
            "dtype": detector.dtype,
            "python_version": sys.version,
            "transformers_version": detector.transformers_version,
            "peft_version": importlib.metadata.version("peft"),
        },
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "complete" if len(successful) == len(validation) else "incomplete"
        ),
        "completed_count": len(successful),
        "failure_count": len(rows) - len(successful),
        **summary,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"CSV: {output_path}")
    if len(successful) != len(validation):
        sys.exit(1)


if __name__ == "__main__":
    main()
