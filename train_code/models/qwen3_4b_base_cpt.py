"""
qwen3_4b_base_cpt.py — Qwen3-4B-Base + Continued Pretraining LoRA Model Wrapper
Loads the base Qwen3-4B-Base model, then merges the continue-pretraining LoRA weights,
so the merged model can serve as a new "base model" for downstream fine-tuning.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List


class Qwen3_4B_Base_CPT:
    """Qwen3-4B-Base + Continued Pretraining LoRA Wrapper

    model_path points to the continue-pretraining LoRA adapter directory (contains adapter_config.json etc.),
    the base model path is passed via base_model_path.
    """

    MODEL_KEY = "qwen3_4b_base_cpt"

    def __init__(self, model_path: str, device: str = "cuda",
                 torch_dtype=torch.float16, base_model_path: str = None):
        """
        Args:
            model_path: path to the continue-pretraining LoRA adapter directory
            device: device
            torch_dtype: model data type
            base_model_path: path to the base Qwen3-4B-Base model (if None, retrieved from MODEL_PATHS)
        """
        self.device = device

        # Get base model path
        if base_model_path is None:
            from config.path import MODEL_PATHS
            base_model_path = MODEL_PATHS["qwen3_4b_base"]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            dtype=torch_dtype,
        )

        # Load continue-pretraining LoRA adapter and merge into base model
        peft_model = PeftModel.from_pretrained(base_model, model_path)
        self.model = peft_model.merge_and_unload()

        self.model = self.model.to(device)
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
