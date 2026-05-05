"""
qwen3_4b_instruct_2507.py — Qwen3-4B-Instruct-2507 Model Wrapper
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


class Qwen3_4B_Instruct_2507:
    """Qwen3-4B-Instruct-2507 LLM Wrapper"""

    MODEL_KEY = "qwen3_4b_instruct_2507"

    def __init__(self, model_path: str, device: str = "cuda",
                 torch_dtype=torch.float16):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch_dtype,
        ).to(device)
        self.model.config.output_hidden_states = True
        self.hidden_dim = self.model.config.hidden_size

    def tokenize(self, texts: List[str]):
        device = next(self.model.parameters()).device
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

    def get_hidden_states(self, inputs) -> torch.Tensor:
        """Return hidden states from the last decoder layer (batch, seq_len, hidden_dim)"""
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1]
