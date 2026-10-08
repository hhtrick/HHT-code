"""
qwen3_4b_base.py — Qwen3-4B-Base Model Wrapper
"""
import torch
from models.llm_io import resolve_model_path, load_llm_tokenizer, load_llm_model
from models.llm_tokenization import tokenize_llm_texts
from typing import List


class Qwen3_4B_Base:
    """Qwen3-4B-Base LLM Wrapper"""

    MODEL_KEY = "qwen3_4b_base"

    def __init__(self, model_path: str, device: str = "cuda",
                 torch_dtype=torch.float16):
        self.device = device
        model_path = resolve_model_path(model_path)
        self.tokenizer = load_llm_tokenizer(model_path)
        self.model = load_llm_model(model_path, torch_dtype).to(device)
        self.model.config.output_hidden_states = True
        self.hidden_dim = self.model.config.hidden_size

    def tokenize(self, texts: List[str]):
        device = next(self.model.parameters()).device
        return tokenize_llm_texts(self.tokenizer, texts).to(device)

    def get_hidden_states(self, inputs) -> torch.Tensor:
        """Return hidden states from the last decoder layer (batch, seq_len, hidden_dim)"""
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        return outputs.hidden_states[-1]
