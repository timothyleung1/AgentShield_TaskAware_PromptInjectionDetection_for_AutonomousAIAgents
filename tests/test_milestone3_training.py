import unittest
from unittest.mock import patch

import pandas as pd

from milestones.milestone3.scripts.compare_lora import (
    compare_predictions,
    calculate_uncertainty,
    validate_matching_runs,
)
from milestones.milestone2.src.detectors.few_shot import DEFAULT_HF_REVISIONS
from milestones.milestone3.src.training import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    build_messages,
    encode_example,
    validate_no_split_overlap,
    validate_split,
)
from milestones.milestone2.src.icl.dataset import (
    DATASET_NAME,
    DATASET_REVISION,
    load_benchmark_split,
)
from milestones.milestone3.scripts.evaluate_lora import (
    build_summary,
    calculate_metrics,
)
from milestones.milestone3.scripts.eval_ood_benchmark import (
    DEFAULT_MANIFEST,
    OOD_STRESS_PROBES,
    build_few_shot_messages,
    compute_metrics as compute_ood_metrics,
)
from milestones.milestone3.src.detectors.lora import LoRADetector, parse_label


class CharacterTokenizer:
    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False
    ):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return list(rendered.encode("utf-8")) if tokenize else rendered


class TrainingDataTests(unittest.TestCase):
    def test_m2_comparator_uses_the_m3_base_model_revision(self):
        self.assertEqual(
            DEFAULT_HF_REVISIONS[BASE_MODEL],
            BASE_MODEL_REVISION,
        )

    def test_paired_icl_lora_metrics_and_attack_families(self):
        icl = pd.DataFrame(
            [
                ["1", 1, 0, "success", "attack"],
                ["2", 1, 1, "success", "attack"],
                ["3", 0, 1, "success", "benign"],
                ["4", 0, 0, "success", "benign"],
            ],
            columns=["id", "actual", "predicted", "status", "attack_family"],
        )
        lora = icl.copy()
        lora["predicted"] = [1, 0, 0, 0]

        result = compare_predictions(icl, lora)

        self.assertEqual(result["paired_errors"]["fixed_by_lora"], 2)
        self.assertEqual(result["paired_errors"]["regressed_by_lora"], 1)
        self.assertEqual(
            result["attack_family_metrics"]["benign"]["delta_lora_minus_icl"][
                "fpr"
            ],
            -0.5,
        )
        self.assertEqual(len(result["changed_predictions"]), 3)

    def test_comparison_requires_frozen_m2_and_lora_configs(self):
        icl = {
            "split": "validation",
            "backend": "huggingface",
            "manifest_name": "random_seed42_8shot",
            "num_shots": 8,
            "prompt_version": "v2",
            "dataset": "benchmark",
            "dataset_revision": "revision",
            "sample_count": 467,
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "effective_prompt_sha256": "prompt-hash",
            "decoding": {"max_new_tokens": 24},
            "runtime": {
                "model_revision": "model-revision",
                "dtype": "torch.bfloat16",
            },
        }
        lora = {
            "split": "validation",
            "prompt_version": "ZERO_SHOT_V2",
            "dataset": "benchmark",
            "dataset_revision": "revision",
            "sample_count": 467,
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "base_model_revision": "model-revision",
            "prompt_sha256": "prompt-hash",
            "decoding": {"max_new_tokens": 24},
            "runtime": {"dtype": "torch.bfloat16"},
        }

        validate_matching_runs(icl, lora)
        lora["dataset_revision"] = "different"
        with self.assertRaisesRegex(ValueError, "different dataset revisions"):
            validate_matching_runs(icl, lora)

    def test_benchmark_loader_fetches_only_the_requested_split(self):
        with (
            patch(
                "huggingface_hub.hf_hub_download",
                return_value="/cache/train.parquet",
            ) as download,
            patch("datasets.load_dataset", return_value="train-data") as load,
        ):
            result = load_benchmark_split("train")

        self.assertEqual(result, "train-data")
        download.assert_called_once_with(
            repo_id=DATASET_NAME,
            filename="data/train.parquet",
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
        load.assert_called_once_with(
            "parquet",
            data_files={"train": "/cache/train.parquet"},
            split="train",
        )

    def test_benchmark_loader_rejects_test_split(self):
        with self.assertRaisesRegex(ValueError, "test split is sealed"):
            load_benchmark_split("test")

    def test_benchmark_loader_requires_explicit_test_access(self):
        with (
            patch(
                "huggingface_hub.hf_hub_download",
                return_value="/cache/test.parquet",
            ) as download,
            patch("datasets.load_dataset", return_value="test-data") as load,
        ):
            result = load_benchmark_split("test", allow_test=True)

        self.assertEqual(result, "test-data")
        download.assert_called_once_with(
            repo_id=DATASET_NAME,
            filename="data/test.parquet",
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
        load.assert_called_once_with(
            "parquet",
            data_files={"test": "/cache/test.parquet"},
            split="test",
        )

    def test_final_comparison_requires_matching_test_hashes(self):
        icl = {
            "split": "test",
            "test_split_accessed": True,
            "backend": "huggingface",
            "manifest_name": "random_seed42_8shot",
            "num_shots": 8,
            "prompt_version": "v2",
            "dataset": "benchmark",
            "dataset_revision": "revision",
            "sample_count": 545,
            "sample_ids_sha256": "same-hash",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "effective_prompt_sha256": "prompt-hash",
            "decoding": {"max_new_tokens": 24},
            "runtime": {
                "model_revision": "model-revision",
                "dtype": "torch.bfloat16",
                "device": "cuda",
            },
        }
        lora = {
            "split": "test",
            "test_split_accessed": True,
            "prompt_version": "ZERO_SHOT_V2",
            "dataset": "benchmark",
            "dataset_revision": "revision",
            "sample_count": 545,
            "sample_ids_sha256": "same-hash",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "base_model_revision": "model-revision",
            "prompt_sha256": "prompt-hash",
            "decoding": {"max_new_tokens": 24},
            "runtime": {"dtype": "torch.bfloat16", "device": "cuda"},
        }
        validate_matching_runs(icl, lora)

        lora["sample_ids_sha256"] = "different-hash"
        with self.assertRaisesRegex(ValueError, "same sample IDs"):
            validate_matching_runs(icl, lora)

    def test_paired_bootstrap_returns_reproducible_metric_intervals(self):
        rows = pd.DataFrame(
            {
                "actual_icl": [1, 1, 0, 0],
                "predicted_icl": [0, 1, 1, 0],
                "predicted_lora": [1, 1, 0, 0],
                "attack_family_icl": [
                    "attack",
                    "attack",
                    "benign",
                    "benign",
                ],
            }
        )

        first = calculate_uncertainty(rows)
        second = calculate_uncertainty(rows)

        self.assertEqual(first["confidence_level"], 0.95)
        self.assertEqual(first["resamples"], 10_000)
        self.assertEqual(
            first["overall"]["f1"]["delta_lora_minus_icl"],
            second["overall"]["f1"]["delta_lora_minus_icl"],
        )
        self.assertEqual(len(first["overall"]["f1"]["icl"]), 2)

    def test_training_target_masks_prompt_tokens(self):
        example = {
            "id": "train-1",
            "text": "A neutral example.",
            "label": 0,
        }
        encoded = encode_example(CharacterTokenizer(), example, 4096)

        self.assertEqual(len(encoded.input_ids), len(encoded.labels))
        prompt = CharacterTokenizer().apply_chat_template(
            build_messages(example["text"]),
            tokenize=True,
            add_generation_prompt=True,
        )
        self.assertEqual(encoded.labels[: len(prompt)], [-100] * len(prompt))
        self.assertTrue(all(token != -100 for token in encoded.labels[len(prompt) :]))

    def test_invalid_labels_are_rejected(self):
        example = {"id": "bad-label", "text": "Text.", "label": True}
        with self.assertRaisesRegex(ValueError, "Invalid label"):
            encode_example(CharacterTokenizer(), example, 4096)

    def test_duplicate_training_text_is_rejected(self):
        rows = [
            {"id": "1", "text": "same text", "label": 0},
            {"id": "2", "text": " same text ", "label": 1},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate train text"):
            validate_split(rows, "train")

    def test_train_validation_overlap_is_rejected(self):
        train = [{"id": "1", "text": "same text", "label": 0}]
        validation = [{"id": "2", "text": " same text ", "label": 0}]
        with self.assertRaisesRegex(ValueError, "exact-text overlap"):
            validate_no_split_overlap(train, validation)

    def test_train_validation_id_overlap_is_rejected(self):
        train = [{"id": "same", "text": "one", "label": 0}]
        validation = [{"id": "same", "text": "two", "label": 1}]
        with self.assertRaisesRegex(ValueError, "ID overlap"):
            validate_no_split_overlap(train, validation)

    def test_evaluation_metrics_and_family_breakdown(self):
        rows = [
            {
                "actual": 1,
                "predicted": 1,
                "status": "success",
                "attack_family": "attack",
                "latency_seconds": 1.0,
                "input_tokens": 10,
                "output_tokens": 4,
            },
            {
                "actual": 0,
                "predicted": 1,
                "status": "success",
                "attack_family": "benign",
                "latency_seconds": 2.0,
                "input_tokens": 20,
                "output_tokens": 4,
            },
            {
                "actual": 0,
                "predicted": 0,
                "status": "success",
                "attack_family": "benign",
                "latency_seconds": 3.0,
                "input_tokens": 30,
                "output_tokens": 4,
            },
        ]

        metrics = calculate_metrics(rows)
        summary = build_summary(rows)
        self.assertEqual(
            (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]),
            (1, 1, 1, 0),
        )
        self.assertEqual(summary["attack_family_metrics"]["benign"]["samples"], 2)
        self.assertAlmostEqual(summary["operational"]["latency_seconds"]["mean"], 2)

    def test_parse_label_validates_strict_json_contract(self):
        self.assertEqual(parse_label('{"label": 0}'), 0)
        self.assertEqual(parse_label(' {"label": 1}\n'), 1)
        for bad in (
            "not json",
            '{"label": true}',
            '{"label": 2}',
            '{"label": 0, "extra": 1}',
        ):
            with self.assertRaises(ValueError):
                parse_label(bad)

    def test_lora_detector_merge_adapter_flag(self):
        from pathlib import Path
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock(pad_token_id=0, eos_token="<eos>")
        mock_base = MagicMock()
        mock_peft = MagicMock()
        mock_merged = MagicMock()
        mock_peft.merge_and_unload.return_value = mock_merged

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=mock_base,
            ),
            patch(
                "peft.PeftModel.from_pretrained",
                return_value=mock_peft,
            ),
        ):
            det_unmerged = LoRADetector(
                Path("/tmp/fake_adapter"), device="cpu", dtype="float32", merge_adapter=False
            )
            self.assertIs(det_unmerged.model, mock_peft)
            mock_peft.merge_and_unload.assert_not_called()

            det_merged = LoRADetector(
                Path("/tmp/fake_adapter"), device="cpu", dtype="float32", merge_adapter=True
            )
            self.assertIs(det_merged.model, mock_merged)
            mock_peft.merge_and_unload.assert_called_once()

    def test_ood_stress_probes_balanced_and_metrics_exact(self):
        self.assertEqual(len(OOD_STRESS_PROBES), 24)
        ids = [p["id"] for p in OOD_STRESS_PROBES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(1 for p in OOD_STRESS_PROBES if p["label"] == 0), 12)
        self.assertEqual(sum(1 for p in OOD_STRESS_PROBES if p["label"] == 1), 12)

        metrics = compute_ood_metrics(
            [
                {"actual": 1, "pred": 1, "latency": 0.4},
                {"actual": 1, "pred": 0, "latency": 0.5},
                {"actual": 0, "pred": 1, "latency": 0.3},
                {"actual": 0, "pred": 0, "latency": 0.2},
            ]
        )
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]), (1, 1, 1, 1))
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["fpr"], 0.5)

    def test_default_few_shot_manifest_builds_8shot_messages(self):
        import json

        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        demos = manifest["examples"]
        self.assertEqual(len(demos), 8)
        messages = build_few_shot_messages("Hello world", demos)
        # 1 system + 8*(user+assistant) + 1 target user = 18 messages
        self.assertEqual(len(messages), 18)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
