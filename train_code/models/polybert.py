"""
polybert.py — PolyBERT (DeBERTa-v2) Structure Encoder Wrapper
"""
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional

from utils import build_ratio_vector


POLYBERT_HIDDEN_DIM = 600  # config.json: hidden_size=600


class PolyBERTEncoder:
    """
    Wrapper for the PolyBERT model to extract structure embeddings from SMILES strings.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        self.hidden_dim = POLYBERT_HIDDEN_DIM

    @torch.no_grad()
    def encode(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Encode a list of SMILES strings and return mean-pooled vectors.
        Args:
            smiles_list: list of SMILES strings
        Returns:
            Tensor of shape (len(smiles_list), hidden_dim)
        """
        if not smiles_list:
            return torch.zeros(1, self.hidden_dim, device=self.device)

        inputs = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # mean pooling over tokens (respect attention mask)
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)
        summed = (hidden_states * mask).sum(dim=1)  # (batch, hidden_dim)
        counts = mask.sum(dim=1).clamp(min=1)       # (batch, 1)
        mean_pooled = summed / counts                # (batch, hidden_dim)

        return mean_pooled

    @torch.no_grad()
    def encode_entry(self, entry: dict, ratio_info: Optional[dict] = None) -> torch.Tensor:
        """
        Extract structure vectors from a data entry.
        Reads monomers[].smiles directly, then concatenates each monomer's structure vector
        with its ratio information vector.
        
        Args:
            entry: data entry
            ratio_info: ratio encoding info dict, containing 'unit_to_onehot' mapping and 'ratio_dim'
        
        Returns:
            Tensor of shape (n_monomers, hidden_dim + ratio_dim) or (n_monomers, hidden_dim)
            where n_monomers >= 1
        """
        chem = entry.get("chemical_composition", {})
        monomers = chem.get("monomers", [])
        is_homo = chem.get("is_homopolymer", False)

        # Collect valid SMILES and corresponding ratio info
        valid_smiles = []
        valid_ratio_vectors = []
        
        for m in monomers:
            smiles = m.get("smiles")
            if smiles and smiles.strip():
                valid_smiles.append(smiles.strip())
                if ratio_info is not None:
                    rv = build_ratio_vector(m, is_homo, ratio_info)
                    valid_ratio_vectors.append(torch.tensor(rv, dtype=torch.float32))

        if not valid_smiles:
            # No valid SMILES, return zeros
            dim = self.hidden_dim
            if ratio_info is not None:
                dim += ratio_info["ratio_dim"]
            return torch.zeros(1, dim, device=self.device)

        # Encode SMILES
        struct_vectors = self.encode(valid_smiles)  # (n, hidden_dim)

        # Concatenate ratio information
        if ratio_info is not None and valid_ratio_vectors:
            ratio_tensor = torch.stack(valid_ratio_vectors).to(self.device)  # (n, ratio_dim)
            struct_vectors = torch.cat([struct_vectors, ratio_tensor], dim=-1)  # (n, hidden_dim + ratio_dim)

        return struct_vectors


