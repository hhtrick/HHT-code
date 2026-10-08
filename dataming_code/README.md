# Literature data-mining pipeline

This directory contains the reusable extraction workflow for polymer property records. It processes main-article PDFs and tables, performs two-stage LLM extraction/review, generates text variants for input ablations, and exports candidate JSON datasets. Run commands from this directory.

## 1. Environment and source files

Use Python 3.10 or later and a MinerU-compatible environment:

```bash
python -m pip install -r requirements.txt
```

MinerU model downloads and GPU requirements depend on the selected backend. The supplied parser script uses the MinerU CLI; the extraction and text-generation scripts use an OpenAI-compatible endpoint.

Create a local `.env` from `.env.example` and provide your own `DASHSCOPE_API_KEY`. Both LLM scripts read this environment variable. The configured endpoint is `https://dashscope.aliyuncs.com/compatible-mode/v1`; change the client settings if using another compatible service. No API key is included.

Place lawfully obtained PDFs in property-specific folders:

```text
dataming_code/
├── Tg/pdf/
├── Tm/pdf/
├── E/pdf/
├── UTS/pdf/
├── eps/pdf/
└── n/pdf/
```

Only create the directories you use. Raw articles and intermediate extraction files are not distributed with the training datasets.

## 2. Pipeline stages

Set `TARGET_DATASETS` consistently at the top of **each** numbered script before running it; the retained working defaults are not identical across all stages.

| Script | Input | Output / function |
|---|---|---|
| `01_pdf_to_md.py` | `{property}/pdf/*.pdf` | `{property}/md/*.md`, parsed by MinerU |
| `02_data_extractor.py` | Markdown article text | Two-stage extraction/review, schema checks and routed JSON output |
| `03_add_title.py` | `{property}_extracted.json` and Markdown | `{property}_extracted_titled.json`, adding the first level-one article heading |
| `04_data_processor.py` | Titled extraction records | `final/{property}.json`, adding two generated-analysis objects |
| `05_export_dataset.py` | `final/{property}.json` | `dataset/{property}.json`, removing the temporary `filename` field |

```bash
python 01_pdf_to_md.py
python 02_data_extractor.py
python 03_add_title.py
python 04_data_processor.py
python 05_export_dataset.py
```

These scripts are configured through their top-level parameters rather than command-line flags.

### PDF parsing

`01_pdf_to_md.py` controls `BACKEND`, batch size, formula/table parsing and MinerU resource settings. Its temporary processing directory is `0/`. It copies extracted Markdown to the selected property directory and maintains the association with the original PDF filename.

### Extraction and review

`02_data_extractor.py` uses extraction prompts and a separate review prompt from `prompt.py`. The retained model settings are `qwen3.5-plus-2026-02-15` for extraction and `glm-5` for review. Configure model names according to the endpoint you use.

The second stage reviews the draft records against the article text. `format_checker.py` checks required fields and JSON types and separates accepted from rejected outputs. `CONCURRENCY`, `MAX_RETRIES`, `MODE` and `IGNORE_PROGRESS` control execution. Production mode uses progress files to support resuming completed articles; test mode processes a small subset. `IGNORE_PROGRESS=True` requests reprocessing and should be set deliberately.

### Titles and generated text

`03_add_title.py` reads the first Markdown level-one heading as a title. It does not independently verify bibliographic metadata; articles with missing or incorrect headings require later checking.

`04_data_processor.py` creates `analysis_with_hierarchical_structure` and `analysis_without_hierarchical_structure`, each containing `descriptive_text` and `reasoning_analysis`. The retained model is `qwen3.5-plus-2026-02-15`. These are generated input variants, not additional experimentally measured facts. The two variants support the training code's input-strategy ablations.

## 3. Record fields

The target properties are `Tg`, `Tm`, `E`, `UTS`, `eps` and `n`. The extraction schema includes:

- `doi`, `title`, `year`: source metadata.
- `chemical_composition`: repeat-unit pSMILES, monomer names/SMILES, composition values and units, additives, and `is_homopolymer`.
- `processing_history`: synthesis, fabrication and relevant measurement context reported in the article.
- `hierarchical_structure`: reported morphology or hierarchical-structure information.
- `properties`: extracted property values and units.
- The two generated-analysis objects added during stage 4.

SMILES and pSMILES are proposed by the LLM. `format_checker.py` validates their field types, not their chemical correctness; it does not run RDKit or establish that a structure matches the source material. Missing values can be represented as `null` according to the prompt/schema rules.

## 4. Export versus the curated benchmark

The numbered pipeline produces candidate records. Unit normalization, RDKit screening, source verification, DOI correction, removal of input leakage, deduplication and the final benchmark partition decisions are separate curation steps. Stage 5 removes `filename`; it does not automatically perform these curation operations.

The final curated data used for training are distributed under `../train_code/dataset`, `../train_code/continue_train_dataset` and `../train_code/val_dataset`. Do not replace those files with a fresh extraction while retaining their pickle indices. LLM outputs and article parsing can vary between runs, so newly mined records need their own quality checks and partitioning.

## 5. Included and excluded files

The release includes the five numbered scripts, `prompt.py`, `format_checker.py`, dependencies and an empty-key `.env.example`. It excludes personal credentials, source PDFs/XML, mined article text, intermediate JSON/progress files, one-time domain splitting scripts and historical analysis outputs. The independent training guide explains how to use the distributed curated datasets.
