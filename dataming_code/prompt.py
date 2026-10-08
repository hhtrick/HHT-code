"""
prompt.py — 数据挖掘流程中所有提示词的集中管理模块。

包含:
- 阶段一: 6 类聚合物属性的数据提取提示词 (EXTRACTION_PROMPTS)
- 阶段二: 数据审查与修正提示词 (REVIEW_PROMPT_TEMPLATE)
"""

# ===========================================================================
# 通用 JSON 模板（用于各属性提取提示词内部引用）
# ===========================================================================
_JSON_TEMPLATE = '''
{{
  "doi": "string or null",
  "year": "integer or null",
  "chemical_composition": {{
    "repeat_unit_psmiles": "string or null",
    "monomers": [
      {{
        "name": "string",
        "smiles": "string or null",
        "ratio_value": "number or null",
        "ratio_unit": "string or null"
      }}
    ],
    "additives": [
      {{
        "name": "string",
        "type": "string (e.g. filler, plasticizer, compatibilizer, metal, crosslinker, initiator, stabilizer, etc.)",
        "amount_value": "number or null",
        "amount_unit": "string or null"
      }}
    ],
    "is_homopolymer": "boolean"
  }},
  "processing_history": "string or null",
  "hierarchical_structure": "string or null",
  "properties": {{
    "{PROP_KEY}": {{
      "value": ["number"],
      "unit": "string or null"
    }}
  }}
}}
'''

# ===========================================================================
# 属性元信息
# ===========================================================================
PROPERTY_META = {
    "Tg": {
        "full_name": "Glass Transition Temperature (Tg)",
        "typical_units": '"°C" or "K"',
        "prop_key": "Tg",
    },
    "Tm": {
        "full_name": "Melting Temperature (Tm)",
        "typical_units": '"°C" or "K"',
        "prop_key": "Tm",
    },
    "n": {
        "full_name": "Refractive Index (n)",
        "typical_units": 'null (dimensionless)',
        "prop_key": "n",
    },
    "eps": {
        "full_name": "Dielectric Constant (ε)",
        "typical_units": 'null (dimensionless)',
        "prop_key": "eps",
    },
    "E": {
        "full_name": "Young's Modulus (E)",
        "typical_units": '"GPa" or "MPa"',
        "prop_key": "E",
    },
    "UTS": {
        "full_name": "Ultimate Tensile Strength (UTS)",
        "typical_units": '"MPa" or "GPa"',
        "prop_key": "UTS",
    },
}

# ===========================================================================
# 属性特定的示例 JSON（嵌入到各提示词中）
# ===========================================================================
_EXAMPLES = {
    "Tg": '''[
  {{
    "doi": "10.1016/j.polymer.2012.01.045",
    "year": 2012,
    "chemical_composition": {{
      "repeat_unit_psmiles": "[*]OC(=O)Oc1ccc(C(C)(C)c2ccc(O*)cc2)cc1",
      "monomers": [
        {{"name": "Bisphenol A", "smiles": "CC(C)(c1ccc(O)cc1)c2ccc(O)cc2", "ratio_value": null, "ratio_unit": null}},
        {{"name": "Phosgene", "smiles": "C(=O)(Cl)Cl", "ratio_value": null, "ratio_unit": null}}
      ],
      "additives": [],
      "is_homopolymer": false
    }},
    "processing_history": "Multilayer co-extrusion technique. PC/PMMA-1024 sample with 1024 layers, average layer thickness 125 nm, composition ratio 50:50 by volume. Tg measured by TMDSC at 2 °C/min heating rate.",
    "hierarchical_structure": "Continuous alternating layers of PC and PMMA with almost uniform thicknesses. Layer thickness 125 nm. Amorphous polymers.",
    "properties": {{
      "Tg": {{
        "value": [143.3],
        "unit": "°C"
      }}
    }}
  }},
  {{
    "doi": "10.1002/macp.200290030",
    "year": 2002,
    "chemical_composition": {{
      "repeat_unit_psmiles": "[*]CC(C1=CC=CC=C1)[*]",
      "monomers": [
        {{"name": "Styrene", "smiles": "C=CC1=CC=CC=C1", "ratio_value": null, "ratio_unit": null}}
      ],
      "additives": [
        {{"name": "Shimalite NAW-101 silicate powder", "type": "filler", "amount_value": null, "amount_unit": null}}
      ],
      "is_homopolymer": true
    }},
    "processing_history": "Atactic monodisperse polystyrene supplied by Polymer Laboratories Ltd. 500 mg PSt dissolved in benzene (40 ml) at room temperature, mixed with Shimalite NAW-101 silicate powder (4 g, 60-80 mesh), dried in vacuum. Tg measured by DSC at 5 °C/min under N2 atmosphere.",
    "hierarchical_structure": "Amorphous polystyrene, monodisperse (Mw/Mn < 1.05-1.07), Mp = 7,600 g/mol",
    "properties": {{
      "Tg": {{
        "value": [88],
        "unit": "°C"
      }}
    }}
  }}
]''',
    "Tm": '''[
  {{
    "doi": "10.1016/j.polymer.2020.01.001",
    "year": 2020,
    "chemical_composition": {{
      "repeat_unit_psmiles": "[*]OC(=O)CCCCC(=O)O[*]",
      "monomers": [
        {{"name": "Adipic acid", "smiles": "OC(=O)CCCCC(=O)O", "ratio_value": 50, "ratio_unit": "mol%"}},
        {{"name": "1,6-hexanediol", "smiles": "OCCCCCCO", "ratio_value": 50, "ratio_unit": "mol%"}}
      ],
      "additives": [],
      "is_homopolymer": false
    }},
    "processing_history": "Polycondensation at 220 °C under vacuum for 6 h with Ti(OBu)4 catalyst. Tm measured by DSC at 10 °C/min heating rate.",
    "hierarchical_structure": "Semicrystalline, crystallinity ~45% by XRD.",
    "properties": {{
      "Tm": {{
        "value": [56.2],
        "unit": "°C"
      }}
    }}
  }}
]''',
    "n": '''[
  {{
    "doi": "10.1016/j.optmat.2019.05.010",
    "year": 2019,
    "chemical_composition": {{
      "repeat_unit_psmiles": "[*]CC(C(=O)OC)[*]",
      "monomers": [
        {{"name": "Methyl methacrylate", "smiles": "CC(=C)C(=O)OC", "ratio_value": null, "ratio_unit": null}}
      ],
      "additives": [],
      "is_homopolymer": true
    }},
    "processing_history": "Bulk polymerization with AIBN at 60 °C for 24 h. Cast into film. Refractive index measured by spectroscopic ellipsometry at 589 nm (sodium D-line), 25 °C.",
    "hierarchical_structure": "Amorphous, optically transparent film, thickness ~200 μm.",
    "properties": {{
      "n": {{
        "value": [1.492],
        "unit": null
      }}
    }}
  }}
]''',
    "eps": '''[
  {{
    "doi": "10.1016/j.polymer.2018.03.020",
    "year": 2018,
    "chemical_composition": {{
      "repeat_unit_psmiles": "[*]CC(F)(F)[*]",
      "monomers": [
        {{"name": "Vinylidene fluoride", "smiles": "FC(F)=C", "ratio_value": null, "ratio_unit": null}}
      ],
      "additives": [
        {{"name": "BaTiO3 nanoparticles", "type": "filler", "amount_value": 10, "amount_unit": "vol%"}}
      ],
      "is_homopolymer": true
    }},
    "processing_history": "Solution casting from DMF, dried at 80 °C for 12 h. Hot-pressed at 200 °C. Dielectric constant measured by impedance spectroscopy at 1 kHz, 25 °C.",
    "hierarchical_structure": "Semicrystalline, β-phase dominant, nanoparticles uniformly dispersed.",
    "properties": {{
      "eps": {{
        "value": [18.5],
        "unit": null
      }}
    }}
  }}
]''',
    "E": '''[
  {{
    "doi": "10.1016/j.compscitech.2019.04.015",
    "year": 2019,
    "chemical_composition": {{
      "repeat_unit_psmiles": "[*]OC(=O)c1ccc(C(=O)O[*])cc1",
      "monomers": [
        {{"name": "Terephthalic acid", "smiles": "OC(=O)c1ccc(C(=O)O)cc1", "ratio_value": 50, "ratio_unit": "mol%"}},
        {{"name": "Ethylene glycol", "smiles": "OCCO", "ratio_value": 50, "ratio_unit": "mol%"}}
      ],
      "additives": [
        {{"name": "Carbon nanotubes", "type": "filler", "amount_value": 1.0, "amount_unit": "wt%"}}
      ],
      "is_homopolymer": false
    }},
    "processing_history": "Melt compounding at 270 °C with twin-screw extruder, injection molded into dog-bone specimens. Young's modulus measured by uniaxial tensile test at 5 mm/min crosshead speed, 23 °C, according to ASTM D638.",
    "hierarchical_structure": "Semicrystalline PET matrix with dispersed CNTs. Crystallinity ~35% by DSC.",
    "properties": {{
      "E": {{
        "value": [3.8],
        "unit": "GPa"
      }}
    }}
  }}
]''',
    "UTS": '''[
  {{
    "doi": "10.1016/j.matdes.2020.108650",
    "year": 2020,
    "chemical_composition": {{
      "repeat_unit_psmiles": "[*]OC(C)(C)c1ccc(Oc2ccc(S(=O)(=O)c3ccc(O[*])cc3)cc2)cc1",
      "monomers": [
        {{"name": "Bisphenol A", "smiles": "CC(C)(c1ccc(O)cc1)c2ccc(O)cc2", "ratio_value": null, "ratio_unit": null}},
        {{"name": "4,4'-Dichlorodiphenyl sulfone", "smiles": "Clc1ccc(S(=O)(=O)c2ccc(Cl)cc2)cc1", "ratio_value": null, "ratio_unit": null}}
      ],
      "additives": [
        {{"name": "Glass fiber", "type": "filler", "amount_value": 30, "amount_unit": "wt%"}}
      ],
      "is_homopolymer": false
    }},
    "processing_history": "Injection molding at 340 °C, mold temperature 150 °C, drying at 120 °C for 4 h before processing. UTS measured by tensile testing at 50 mm/min, 23 °C, ISO 527.",
    "hierarchical_structure": "Amorphous matrix with aligned glass fibers, fiber length 200-400 μm.",
    "properties": {{
      "UTS": {{
        "value": [145],
        "unit": "MPa"
      }}
    }}
  }}
]''',
}


def _build_extraction_prompt(prop_key: str) -> str:
    """构建指定属性的阶段一数据提取提示词。"""
    meta = PROPERTY_META[prop_key]
    full_name = meta["full_name"]
    typical_units = meta["typical_units"]
    template = _JSON_TEMPLATE.replace("{PROP_KEY}", prop_key)
    example = _EXAMPLES[prop_key]

    return f'''# Role
You are an elite Materials Informatics and Polymer Chemistry Expert. Your task is to accurately, comprehensively, and systematically extract polymer property data from provided scientific literature and output it in a strictly defined structured JSON format.

# Task
Read the provided Markdown text of a scientific paper and identify **ALL distinct polymer samples** synthesized, prepared, or characterized in the study. For each unique sample, extract its publication year, DOI, chemical composition, processing history, hierarchical structure, and specific properties.

# Extraction Rules (CRITICAL)

1. **Definition of a Distinct Sample (No Merging)**:
   - **CRITICAL**: Any variation in `chemical_composition` (e.g., different monomers or different monomer ratios), `processing_history` (e.g., changes in solvent, reaction time, annealing temperature, film-casting method), `hierarchical_structure`, or **characterization/testing method** constitutes a **NEW and DISTINCT polymer sample**.
   - You MUST create a separate JSON data entry (object) for each distinct sample.
   - **NEVER** merge the property values of different samples into a "min-max" range. A range should ONLY be used if the paper explicitly reports a single sample's property as a range or with an error bar.
   - **CRITICAL — ONE VALUE PER ENTRY**: If the same physical sample is characterized by **different instruments or under different testing conditions** (e.g., DSC vs. DMA for Tg, or different frequencies/heating rates), each measurement MUST produce a **separate entry**. Include the characterization method, instrument, and testing conditions (e.g., heating rate, frequency, wavelength, strain rate) in the `processing_history` field. This ensures each entry contains only **one** target property value.

2. **Paper Identification**:
   - `doi`: Extract the DOI of the paper. Output `null` if not found.
   - `year`: Extract the publication year of the paper as an integer (e.g., 2023). Output `null` if not found.

3. **Chemical Composition Extraction**:
   - `repeat_unit_psmiles`: Deduce or extract the Polymer SMILES (PSMILES) of the repeat unit based on the text. If impossible to determine, output `null`. NEVER hallucinate.
   - `monomers`: Extract ALL monomers used to synthesize the polymer. For each monomer, provide its `smiles`.
   - `ratio_value` and `ratio_unit`: Extract the stoichiometric ratio or feed ratio for each monomer exactly as stated (e.g., unit can be "mol%", "wt%"). If the ratio is not mentioned, output `null`.
   - `is_homopolymer`: If the polymer is synthesized via homopolymerization of a single monomer, set this boolean field to `true`, provide the single monomer's SMILES, and set the ratios to `null`.
   - `additives`: Extract ALL non-monomer components present in the polymer system, including but not limited to: fillers (e.g., silica, clay, carbon nanotubes), plasticizers, compatibilizers, metals or metal oxides, crosslinkers, initiators, stabilizers, flame retardants, nucleating agents, etc. For each additive, provide its `name`, `type` (category), `amount_value`, and `amount_unit`. If no additives are present, output an empty array `[]`.

4. **Processing & Structure Extraction**:
   - `processing_history`: Extract a concise but complete text description of: (a) the synthesis operations, solvents, temperatures, and fabrication history (e.g., spin-coating, hot-pressing, annealing); AND (b) the **characterization method, instrument, and testing conditions** used to measure the target property (e.g., "Tg measured by DSC at 10 °C/min under N2", "Refractive index measured by ellipsometry at 589 nm, 25 °C").
   - `hierarchical_structure`: Extract text descriptions or specific characterization values related to multi-level structures (e.g., crystallinity, phase separation, morphology, free volume, XRD d-spacing).

5. **Property Data Extraction**:
   - Your SOLE focus for properties is to extract the **{full_name}**. Ignore other properties.
   - **CRITICAL — NO EMPTY ENTRIES**: Do NOT create an entry for a sample if it does NOT have a reported {full_name} value. Only record samples that have explicit numerical {full_name} data.
   - **NO FORCED EXTRACTION**: If the paper does not contain any {full_name} data at all, or is entirely unrelated to polymer {full_name}, output an empty JSON array `[]`. Do NOT fabricate or force-extract data that does not exist in the paper.
   - The data structure MUST strictly contain two fields: `value` (array) and `unit` (string or null).
   - `value`: An array of numerical values. In most cases this should be a single-element array (e.g., `[250.5]`). Only use multiple elements if the paper explicitly reports the same measurement as a range or with error bounds.
   - `unit`: {typical_units}. Extract as a single string. Use `null` if the property is dimensionless.

# Output Format
You MUST output **ONLY** a valid JSON array of objects. Do NOT include any Markdown formatting blocks (e.g., ```json), introductory text, or explanations. Just the raw JSON string.

Template:
{template}

JSON Schema Example:
{example}

# Input Text
Here is the text of the literature to process:
'''


# ===========================================================================
# 阶段一提取提示词字典
# ===========================================================================
EXTRACTION_PROMPTS = {k: _build_extraction_prompt(k) for k in PROPERTY_META}


def get_extraction_prompt(prop_key: str, md_text: str) -> str:
    """返回拼接了文献 Markdown 内容的完整阶段一提示词。"""
    base = EXTRACTION_PROMPTS[prop_key]
    return base + "\n" + md_text


# ===========================================================================
# 阶段二：数据审查与修正提示词
# ===========================================================================
_REVIEW_RULES_BRIEF = '''
Key extraction rules for reference:
- Each entry corresponds to ONE distinct polymer sample with ONE target property value.
- Different characterization methods or testing conditions for the same sample MUST be separate entries (method/condition info is included in `processing_history`).
- `chemical_composition` must include `monomers` (with SMILES) and `additives` (fillers, plasticizers, metals, etc.); `is_homopolymer` is boolean.
- `processing_history` must contain synthesis details AND characterization method/conditions.
- `properties` contains ONLY `value` (array) and `unit` (string or null). The `value` array should typically have one element.
- Entries without a reported target property value should NOT exist.
- SMILES strings must be chemically valid. Do NOT hallucinate structures.
'''

REVIEW_PROMPT_TEMPLATE = '''# Role
You are a senior data quality reviewer specializing in polymer informatics. Your task is to verify the accuracy and completeness of structured JSON data extracted from a scientific paper.

# Task
Below is a JSON dataset extracted from the provided Markdown text of a scientific paper, along with the extraction rules that were used. Your job is to carefully cross-check the extracted data against the original text and the rules.

# Instructions
1. Verify that ALL distinct polymer samples with reported {PROP_FULL_NAME} values have been captured — no samples should be missing.
2. Verify that each entry's `doi`, `year`, `chemical_composition` (including `monomers` SMILES, `additives`), `processing_history`, `hierarchical_structure`, and `properties` are accurate and correspond to the original text.
3. Verify that the characterization method and testing conditions are included in `processing_history`, NOT in `properties`.
4. Verify that entries without an explicit target property value do NOT exist.
5. Verify that different measurement methods for the same sample are in separate entries.
6. If the extracted data is an empty array `[]`, verify whether the paper genuinely contains no relevant {PROP_FULL_NAME} data. If confirmed, output `[]`. Do NOT force extraction of non-existent data.
7. If the extracted data is entirely correct (including correctly empty), output an empty JSON array: `[]`
8. If there are ANY errors (missing entries, wrong values, incorrect SMILES, missing additives, etc.), output the **complete corrected dataset** as a JSON array — include ALL entries, both the corrected ones and the ones that were already correct.

# Output Format
You MUST output **ONLY** a valid JSON array. Do NOT include any Markdown formatting blocks (e.g., ```json), introductory text, or explanations.
- If data is correct: output `[]`
- If data has errors: output the full corrected JSON array with ALL entries.

# Extraction Rules Summary
{RULES}

# Extracted Data to Review
{EXTRACTED_DATA}

# Original Paper Text
{MD_TEXT}
'''


def get_review_prompt(prop_key: str, extracted_json_str: str, md_text: str) -> str:
    """返回拼接了提取数据和原文的阶段二审查提示词。"""
    meta = PROPERTY_META[prop_key]
    return REVIEW_PROMPT_TEMPLATE.replace(
        "{PROP_FULL_NAME}", meta["full_name"]
    ).replace(
        "{RULES}", _REVIEW_RULES_BRIEF
    ).replace(
        "{EXTRACTED_DATA}", extracted_json_str
    ).replace(
        "{MD_TEXT}", md_text
    )
