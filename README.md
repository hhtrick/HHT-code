# HHT — Holistic History-aware Text for Polymer Property Prediction

HHT represents polymer composition and processing history as text, encodes it with a pretrained language model, and predicts properties using learned pooling and an MLP regression head. This repository contains the literature-mining pipeline, training code, curated datasets and fixed evaluation partitions for six polymer properties and two continual-training domains.

## Repository structure

```text
.
├── train_code/       # Training, embedding caches, inference, curated data and split indices
│   └── README.md
├── dataming_code/    # PDF → Markdown → extraction/review → text generation → JSON export
│   └── README.md
└── README.md
```

See the [training guide](train_code/README.md) and [data-mining guide](dataming_code/README.md) for configuration and execution instructions. Run commands from the corresponding subdirectory.

## Models and experiments

- **HHT:** frozen LLM representations with sigmoid-gated pooling and an MLP. The standard ChemDFM configuration samples input Types 2 and 4 during training and uses Type 2 for validation and testing.
- **HHT-C:** complete chemical-composition information without processing history, implemented as input Type 9.
- **Structural comparisons:** adapted PolyBERT and PerioGT, plus LLM-struct with reduced structural input fields.
- **Evaluation:** fixed entry-level random and DOI-based article-level splits, three training seeds, nested training fractions, joint training and domain-specific continual training.
- **Article-level baseline:** training-article means or medians, with the corresponding global training statistic for unseen articles.

The frozen LLM path caches final-layer hidden states. Pooling and regression are trained from these caches; input prompts are used without a chat-template wrapper. PolyBERT and PerioGT retain their own preprocessing and training behavior. PerioGT's graph backbone is trainable.

## Included datasets

The six main datasets contain **8,831 records from 1,277 distinct source articles**. Article counts below are property-specific and therefore should not be summed to obtain the distinct total.

| Property | Unit | Records | Source articles |
|---|---|---:|---:|
| Glass transition temperature (`Tg`) | °C | 1,718 | 209 |
| Melting temperature (`Tm`) | °C | 1,348 | 220 |
| Young's modulus (`E`) | GPa | 1,302 | 194 |
| Ultimate tensile strength (`UTS`) | MPa | 1,877 | 256 |
| Relative dielectric constant (`eps`) | dimensionless | 1,286 | 241 |
| Refractive index (`n`) | dimensionless | 1,300 | 161 |

The dielectric prediction task is restricted to `0 < eps <= 1000`. The additional domains contain 879 epoxy-modulus records and 73 polymeric phase-change-material melting-temperature records. `E` and `E_article` in the continual-training folders are alternative partitions of the same epoxy corpus, not two independent datasets.

[mined_publications.csv](mined_publications.csv) lists the source DOIs for all main and domain datasets, including held-out domain records. Its two columns are `doi` (DOI) and `property` (training target). Each DOI–property pair appears once; `E_article` is counted as `E`. The file contains 1,431 pairs representing 1,426 distinct articles. The larger article total includes the domain corpora; the six main datasets alone contain 1,277 distinct articles.

Dataset JSON files and their accompanying pickle indices form one versioned unit. Do not reorder records or substitute newly mined records while retaining these indices. The published training inputs also include LLM-generated descriptions and reasoning for input-strategy ablations; those fields should not be treated as independently verified source facts.

## Setup

Use separate environments for training and MinerU if their dependencies conflict. Training requires a compatible PyTorch/CUDA installation and downloaded pretrained weights; data mining requires MinerU and an OpenAI-compatible API service.

```bash
cd train_code
python -m pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build appropriate for your machine before the training dependencies. FlashAttention is optional; the loader can fall back to PyTorch SDPA. Pretrained model weights, embedding caches, GPU logs and raw literature files are not included.

For data mining:

```bash
cd dataming_code
python -m pip install -r requirements.txt
```

Detailed model paths, API configuration, input choices, split protocols and domain settings are documented in the two subdirectory guides. Configuration files expose research settings; the training guide identifies the settings needed to reproduce the reported experiments, including target transformation and epoch budgets.

## Scope of the release

The released datasets include source checking, corrections, exclusions and deduplication performed after automated extraction. Running the mining pipeline on new PDFs produces new candidate records; it does not automatically reproduce those subsequent curation decisions or the supplied benchmark partitions.

The adapted PerioGT implementation is contained in `train_code/models/PerioGT`. A separate checkout of the upstream reference code is not required. One-time audit/replay scripts and experiment output archives are not part of this release.

This layout follows the [HHT-code repository](https://github.com/hhtrick/HHT-code), with documentation updated for the included code and revised datasets.
