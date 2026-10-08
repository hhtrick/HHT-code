"""
prompt.py — Manage 9 semantic templates and the standalone LLM structural input
"""
import json
import random
from typing import List, Dict, Any, Optional, Union


STRUCT_INPUT = "struct"


def _extract_chemical_composition(entry: Dict[str, Any]) -> Dict[str, Any]:
    return entry.get("chemical_composition", {})


def _extract_processing_history(entry: Dict[str, Any]) -> Optional[str]:
    return entry.get("processing_history", None)


def _extract_hierarchical_structure(entry: Dict[str, Any]) -> Optional[str]:
    return entry.get("hierarchical_structure", None)


def _get_unit(entry: Dict[str, Any], target_property: str) -> str:
    props = entry.get("properties", {})
    prop_data = props.get(target_property, {})
    return prop_data.get("unit") or ""


# ======================== 9 Prompt Templates ========================

def _build_type1(entry: Dict[str, Any], target_property: str, unit: str) -> str:
    """chemical_composition + processing_history + hierarchical_structure"""
    data = {
        "chemical_composition": _extract_chemical_composition(entry),
        "processing_history": _extract_processing_history(entry),
        "hierarchical_structure": _extract_hierarchical_structure(entry),
    }
    json_str = json.dumps(data, ensure_ascii=False)
    return f"Here is the data of a polymer, {json_str}, please predict {target_property} based on the data, in units of {unit}"


def _build_type2(entry: Dict[str, Any], target_property: str, unit: str) -> str:
    """chemical_composition + processing_history（no hierarchical_structure）"""
    data = {
        "chemical_composition": _extract_chemical_composition(entry),
        "processing_history": _extract_processing_history(entry),
    }
    json_str = json.dumps(data, ensure_ascii=False)
    return f"Here is the data of a polymer, {json_str}, please predict {target_property} based on the data, in units of {unit}"


def _build_type3(entry: Dict[str, Any], target_property: str, unit: str) -> str:
    """analysis_with_hierarchical_structure → descriptive_text"""
    text = entry.get("analysis_with_hierarchical_structure", {}).get("descriptive_text", "")
    return f"Here is a description of a polymer: {text}. Please predict {target_property} based on the description, in units of {unit}"


def _build_type4(entry: Dict[str, Any], target_property: str, unit: str) -> str:
    """analysis_without_hierarchical_structure → descriptive_text"""
    text = entry.get("analysis_without_hierarchical_structure", {}).get("descriptive_text", "")
    return f"Here is a description of a polymer: {text}. Please predict {target_property} based on the description, in units of {unit}"


def _build_type5(entry: Dict[str, Any], target_property: str, unit: str) -> str:
    """analysis_with_hierarchical_structure → reasoning_analysis"""
    text = entry.get("analysis_with_hierarchical_structure", {}).get("reasoning_analysis", "")
    return f"Here is a reasoning analysis of a polymer: {text}. Please predict {target_property} based on the analysis, in units of {unit}"


def _build_type6(entry: Dict[str, Any], target_property: str, unit: str) -> str:
    """analysis_without_hierarchical_structure → reasoning_analysis"""
    text = entry.get("analysis_without_hierarchical_structure", {}).get("reasoning_analysis", "")
    return f"Here is a reasoning analysis of a polymer: {text}. Please predict {target_property} based on the analysis, in units of {unit}"


def _build_type7(entry: Dict[str, Any], target_property: str, unit: str) -> str:
    """analysis_with_hierarchical_structure → descriptive_text + reasoning_analysis"""
    analysis = entry.get("analysis_with_hierarchical_structure", {})
    desc = analysis.get("descriptive_text", "")
    reason = analysis.get("reasoning_analysis", "")
    return (f"Here is a description and reasoning analysis of a polymer: {desc} {reason}. "
            f"Please predict {target_property} based on the above content, in units of {unit}")


def _build_type8(entry: Dict[str, Any], target_property: str, unit: str) -> str:
    """analysis_without_hierarchical_structure → descriptive_text + reasoning_analysis"""
    analysis = entry.get("analysis_without_hierarchical_structure", {})
    desc = analysis.get("descriptive_text", "")
    reason = analysis.get("reasoning_analysis", "")
    return (f"Here is a description and reasoning analysis of a polymer: {desc} {reason}. "
            f"Please predict {target_property} based on the above content, in units of {unit}")


def _build_type9(entry: Dict[str, Any], target_property: str, unit: str) -> str:
    """chemical_composition only"""
    data = {
        "chemical_composition": _extract_chemical_composition(entry),
    }
    json_str = json.dumps(data, ensure_ascii=False)
    return f"Here is the data of a polymer, {json_str}, please predict {target_property} based on the data, in units of {unit}"


def _build_struct(entry: Dict[str, Any], target_property: str, unit: str) -> str:
    """Shared structural fields only; preserve source values and monomer order.

    This aligns the supplied fields, not each encoder's preprocessing: PolyBERT
    skips empty SMILES, PerioGT uses two slots, and both override homopolymer
    ratios. Names, pSMILES, additives and experimental history are excluded.
    """
    chem = _extract_chemical_composition(entry)
    data = {
        "chemical_composition": {
            "monomers": [
                {key: monomer.get(key) for key in ("smiles", "ratio_value", "ratio_unit")}
                for monomer in chem.get("monomers", [])
            ],
            "is_homopolymer": chem.get("is_homopolymer", False),
        },
    }
    json_str = json.dumps(data, ensure_ascii=False)
    return f"Here is the data of a polymer, {json_str}, please predict {target_property} based on the data, in units of {unit}"


# Template mapping (1-9 are concrete templates; type 10 "dynamic prompt sampling" is implemented by INPUT_CONTENT containing multiple types)
PROMPT_BUILDERS = {
    1: _build_type1,
    2: _build_type2,
    3: _build_type3,
    4: _build_type4,
    5: _build_type5,
    6: _build_type6,
    7: _build_type7,
    8: _build_type8,
    9: _build_type9,
}


def build_prompt(entry: Dict[str, Any], target_property: str,
                 input_type: Union[int, str]) -> str:
    """
    Build a semantic prompt (1-9) or the named standalone structural prompt.
    """
    unit = _get_unit(entry, target_property)
    builder = _build_struct if input_type == STRUCT_INPUT else PROMPT_BUILDERS[input_type]
    return builder(entry, target_property, unit)


def build_prompt_random(entry: Dict[str, Any], target_property: str,
                        input_content: List[int]) -> str:
    """
    Randomly pick one input_type from the input_content list to build the prompt (type-10 dynamic augmentation mode).
    """
    chosen = random.choice(input_content)
    return build_prompt(entry, target_property, chosen)
