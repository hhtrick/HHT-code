# Data Mining Pipeline

Automated pipeline for extracting polymer property data from scientific literature PDFs.

## Pipeline Overview

```
PDF → Markdown (MinerU) → Extraction (LLM Stage 1) → Review (LLM Stage 2) → Validation → Analysis (LLM) → Final Dataset
```

## Scripts

### `01_pdf_to_md.py` — PDF to Markdown Conversion
Converts PDF articles to Markdown using [MinerU](https://github.com/opendatalab/MinerU). Processes PDF files in batch mode with checkpoint/resume support. Configured for GPU acceleration (RTX 5090) with the `hybrid-auto-engine` backend.

- **Input**: PDF files under `{dataset}/pdf/`
- **Output**: Markdown files under `{dataset}/md/`
- **Key features**: batch processing, checkpoint resume, short-name temp directories to avoid Windows path length issues, GPU memory cleanup between batches

  > **Note on `BATCH_SIZE`**: In practice, PDF conversion via MinerU is very fast, and using a smaller `BATCH_SIZE` can trigger issues (e.g., GPU out-of-memory between batches). Setting `BATCH_SIZE` to a value greater than or equal to the total number of PDFs to process allows smooth, uninterrupted conversion. This parameter has been retained to faithfully reflect the original processing workflow, but users should be aware of this behavior.

### `02_data_extractor.py` — Two-Stage LLM Extraction
Main extraction pipeline based on two LLM calls:

1. **Stage 1** (`qwen3.5-plus-2026-02-15`): Extracts structured JSON from Markdown — polymer samples with chemical composition, processing history, and target property values.
2. **Stage 2** (`glm-5`): Reviews Stage 1 output for accuracy and completeness. Returns empty `[]` if correct, or the full corrected dataset if errors found.

Both models use thinking mode (`enable_thinking: True`) via Alibaba Bailian API.

- **Input**: Markdown files under `{dataset}/md/`
- **Output**: `{dataset}_extracted.json` (valid entries), `{dataset}_format_errors.json` (invalid entries)
- **Key features**: async concurrency, checkpoint resume via progress files, test mode (3 random files + reasoning chains saved), append-mode writing to avoid data loss

### `03_add_title.py` — Add Paper Titles
Extracts the first-level heading (paper title) from each source Markdown file and inserts it as a `title` field in the JSON entries (directly below the `doi` field).

- **Input**: `{dataset}_extracted.json` + source Markdown files
- **Output**: `{dataset}_extracted_titled.json`

### `04_data_processor.py` — LLM-Based Data Analysis
Generates natural language descriptions and scientific reasoning analysis for each polymer sample using `qwen3.5-plus-2026-02-15`. Produces two versions per sample:

- **With hierarchical structure**: Synthesizes chemical composition, processing history, and hierarchical structure into fluent text + qualitative reasoning about property influences.
- **Without hierarchical structure**: Same but omitting hierarchical structure information.

- **Input**: `{dataset}_extracted_titled.json`
- **Output**: `final/{dataset}.json` (with `analysis_with_hierarchical_structure` and `analysis_without_hierarchical_structure` fields)
- **Key features**: async concurrency, checkpoint resume, writing results after each item to avoid data loss

### `05_export_dataset.py` — Export Final Dataset
Removes internal fields (e.g., `filename`) and exports the final dataset to `dataset/{dataset}.json`.

- **Input**: `final/{dataset}.json`
- **Output**: `dataset/{dataset}.json`

### `format_checker.py` — JSON Schema Validation
Validates extracted JSON entries against the expected schema (required fields, correct types for `doi`, `year`, `chemical_composition`, `monomers`, `additives`, `processing_history`, `hierarchical_structure`, `properties`). Routes valid entries to the main dataset file and invalid ones to an error file for manual inspection.

### `prompt.py` — Prompt Management
Centralized repository for all LLM prompts used in the extraction pipeline:

- Stage 1 extraction prompts for 6 target properties (Tg, Tm, n, eps, E, UTS)
- Stage 2 review prompt template
- Property metadata dictionary (full names, typical units)

## Usage Workflow

1. Place PDF articles in `{dataset}/pdf/` (e.g., `Tg/pdf/`)
2. Run `01_pdf_to_md.py` to convert PDFs to Markdown
3. Run `02_data_extractor.py` to extract structured JSON data
4. Run `03_add_title.py` to add paper titles
5. Run `04_data_processor.py` to generate natural language descriptions and reasoning
6. Run `05_export_dataset.py` to produce the final dataset

## Configuration

Each script has a configuration section at the top with parameters such as:

- `TARGET_DATASETS`: list of dataset names to process (`["Tg", "Tm", "n", "eps", "E", "UTS"]`)
- `MODE`: `"test"` (3 samples + reasoning chains) or `"prod"` (full processing)
- `CONCURRENCY`: async concurrency level
- `MAX_RETRIES`: maximum API call retries

API keys are loaded from a `.env` file (`DASHSCOPE_API_KEY`).

## Data Schema

Each extracted polymer sample follows a structured JSON format:

```json
{
  "doi": "string or null",
  "title": "string or null",
  "year": "integer or null",
  "filename": "source filename (removed in final export)",
  "chemical_composition": {
    "repeat_unit_psmiles": "string or null",
    "monomers": [{"name": "...", "smiles": "...", "ratio_value": null, "ratio_unit": null}],
    "additives": [{"name": "...", "type": "...", "amount_value": null, "amount_unit": null}],
    "is_homopolymer": false
  },
  "processing_history": "string or null",
  "hierarchical_structure": "string or null",
  "properties": {
    "Tg": {"value": [143.3], "unit": "°C"}
  }
}
```

After `04_data_processor.py`, two additional fields are appended:
- `analysis_with_hierarchical_structure`: `{"descriptive_text": "...", "reasoning_analysis": "..."}`
- `analysis_without_hierarchical_structure`: `{"descriptive_text": "...", "reasoning_analysis": "..."}`

## Notes

- **RDKit SMILES validation**: After the dataset is fully collected, SMILES strings are validated using RDKit via terminal commands. This step is simple and performed outside the pipeline scripts, so no dedicated code is included.
- **Unit harmonization**: After data extraction, unit unification (e.g., converting all temperatures to °C, all moduli to GPa) is also performed via terminal commands on the collected dataset, so there is no dedicated code for this step.
- The pipeline supports 6 polymer properties: Glass Transition Temperature (Tg), Melting Temperature (Tm), Refractive Index (n), Dielectric Constant (eps), Young's Modulus (E), and Ultimate Tensile Strength (UTS).
