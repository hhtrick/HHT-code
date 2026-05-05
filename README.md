# HHT — Holistic History-aware Text for Polymer Property Prediction

A framework that encodes polymer composition, synthesis conditions, and processing history as natural language and uses frozen pretrained large language models (LLMs) for property regression. The project spans a complete workflow from automated literature data mining to property prediction.

## Key Features

- **Automated Literature Mining**: PDF-to-structured-data pipeline using MinerU and LLM-based extraction/review
- **Text-Based Encoding**: Preserves experimental context (synthesis, processing, additives) that structure-only encoders discard
- **Frozen LLM Regression**: Only pooling layers and MLP heads are trained; LLM embeddings are cached in advance for resource efficiency
- **Hybrid Encoding**: Optional fusion of semantic (LLM) and structural (PolyBERT / PerioGT / LLM-struct) information streams via learned gating
- **Article Memory Analysis**: Quantifies and mitigates article-level clustering effects in literature-mined datasets with dedicated baseline and post-hoc metrics
- **6 Polymer Properties**: Glass Transition Temperature (Tg), Melting Temperature (Tm), Refractive Index (n), Dielectric Constant (eps), Young's Modulus (E), Ultimate Tensile Strength (UTS)

## Repository Structure

```
├── dataming_code/     # Data mining pipeline (PDF → Markdown → JSON extraction → dataset export)
│   └── README.md      # Detailed documentation
├── train_code/        # Model training framework (LLM + structural encoders, grid search, inference)
│   └── README.md      # Detailed documentation
└── README.md          # This file
```

See the README files in each subdirectory for detailed usage instructions.

## Requirements

### Data Mining (`dataming_code/`)
- MinerU (PDF parsing)
- OpenAI-compatible API access (Alibaba Bailian)
- Python 3.10+

### Training (`train_code/`)
See `train_code/requirements.txt` for full dependencies. Key packages:
- PyTorch + PyTorch Lightning
- Transformers + PEFT
- Ray Tune (hyperparameter search)
- PyTorch Geometric (PerioGT)
- RDKit (SMILES validation)
