import json
import time
from dataclasses import dataclass
from pathlib import Path

from milestones.milestone3.src.training import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    build_messages,
)


@dataclass(frozen=True)
class LoRAPrediction:
    label: int
    raw_output: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


def parse_label(raw_output: str) -> int:
    try:
        result = json.loads(raw_output.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON model output: {raw_output!r}") from exc

    if not isinstance(result, dict) or set(result) != {"label"}:
        raise ValueError(f"Output does not match {{'label': 0|1}}: {raw_output!r}")
    label = result["label"]
    if isinstance(label, bool) or label not in (0, 1):
        raise ValueError(f"Invalid classification label: {label!r}")
    return int(label)


class LoRADetector:
    def __init__(
        self,
        adapter_path: Path,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 24,
        merge_adapter: bool = False,
    ):
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer

        self.torch = torch
        self.transformers_version = transformers.__version__
        self.adapter_path = Path(adapter_path)
        self.max_new_tokens = max_new_tokens
        self.merge_adapter = merge_adapter

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device: {device}")
        self.device = device

        dtype_map = {
            "auto": None,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        selected_dtype = dtype_map[dtype]
        if selected_dtype is None:
            if device == "cuda" and torch.cuda.is_bf16_supported():
                selected_dtype = torch.bfloat16
            elif device in {"cuda", "mps"}:
                selected_dtype = torch.float16
            else:
                selected_dtype = torch.float32
        if device == "cpu" and selected_dtype != torch.float32:
            raise ValueError("Use float32 for CPU inference.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, revision=BASE_MODEL_REVISION
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            revision=BASE_MODEL_REVISION,
            torch_dtype=selected_dtype,
        )
        peft_model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model = (
            peft_model.merge_and_unload() if self.merge_adapter else peft_model
        )
        self.model.to(device)
        self.model.eval()
        self.dtype = str(selected_dtype)

    def _sync(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def predict(self, text: str) -> LoRAPrediction:
        prompt = self.tokenizer.apply_chat_template(
            build_messages(text),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_tokens = int(input_ids.shape[-1])

        self._sync()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        self._sync()
        latency = time.perf_counter() - started

        completion_ids = output_ids[0, input_tokens:]
        output_tokens = int(completion_ids.shape[-1])
        raw_output = self.tokenizer.decode(
            completion_ids, skip_special_tokens=True
        ).strip()
        return LoRAPrediction(
            label=parse_label(raw_output),
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
        )
