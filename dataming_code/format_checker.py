"""
format_checker.py — JSON 格式验证与分流模块。

验证从两阶段 LLM 管道中提取的聚合物属性数据是否严格满足
预定义的 JSON Schema，并将合格数据与不合格数据分流保存。
"""

import json
import os
from typing import Tuple


def _validate_monomer(monomer: dict) -> list:
    """验证单个 monomer 对象。"""
    errors = []
    if not isinstance(monomer, dict):
        return ["monomer entry is not a dict"]
    if "name" not in monomer or not isinstance(monomer.get("name"), str):
        errors.append("monomer missing valid 'name' (string)")
    if "smiles" not in monomer:
        errors.append("monomer missing 'smiles' field")
    elif monomer["smiles"] is not None and not isinstance(monomer["smiles"], str):
        errors.append("monomer 'smiles' must be string or null")
    for field in ("ratio_value", "ratio_unit"):
        if field not in monomer:
            errors.append(f"monomer missing '{field}' field")
    return errors


def _validate_additive(additive: dict) -> list:
    """验证单个 additive 对象。"""
    errors = []
    if not isinstance(additive, dict):
        return ["additive entry is not a dict"]
    if "name" not in additive or not isinstance(additive.get("name"), str):
        errors.append("additive missing valid 'name' (string)")
    if "type" not in additive or not isinstance(additive.get("type"), str):
        errors.append("additive missing valid 'type' (string)")
    return errors


def validate_entry(entry: dict, prop_key: str) -> Tuple[bool, list]:
    """
    验证单个数据条目是否满足格式要求。

    Args:
        entry: 数据条目（dict）
        prop_key: 目标属性键名（"Tg", "Tm", "n", "eps", "E", "UTS"）

    Returns:
        (is_valid, errors): 是否合格，以及错误信息列表
    """
    errors = []

    # 顶层字段检查
    if not isinstance(entry, dict):
        return False, ["entry is not a dict"]

    # doi
    if "doi" not in entry:
        errors.append("missing 'doi' field")
    elif entry["doi"] is not None and not isinstance(entry["doi"], str):
        errors.append("'doi' must be string or null")

    # year
    if "year" not in entry:
        errors.append("missing 'year' field")
    elif entry["year"] is not None and not isinstance(entry["year"], (int, float)):
        errors.append("'year' must be integer or null")

    # chemical_composition
    cc = entry.get("chemical_composition")
    if not isinstance(cc, dict):
        errors.append("missing or invalid 'chemical_composition' (must be dict)")
    else:
        # repeat_unit_psmiles
        if "repeat_unit_psmiles" not in cc:
            errors.append("chemical_composition missing 'repeat_unit_psmiles'")
        elif cc["repeat_unit_psmiles"] is not None and not isinstance(cc["repeat_unit_psmiles"], str):
            errors.append("'repeat_unit_psmiles' must be string or null")

        # monomers
        monomers = cc.get("monomers")
        if not isinstance(monomers, list):
            errors.append("chemical_composition missing or invalid 'monomers' (must be array)")
        else:
            for i, m in enumerate(monomers):
                for e in _validate_monomer(m):
                    errors.append(f"monomers[{i}]: {e}")

        # additives
        additives = cc.get("additives")
        if additives is None:
            # 允许缺失，视为空数组
            pass
        elif not isinstance(additives, list):
            errors.append("'additives' must be array or null")
        else:
            for i, a in enumerate(additives):
                for e in _validate_additive(a):
                    errors.append(f"additives[{i}]: {e}")

        # is_homopolymer
        if "is_homopolymer" not in cc:
            errors.append("chemical_composition missing 'is_homopolymer'")
        elif not isinstance(cc.get("is_homopolymer"), bool):
            errors.append("'is_homopolymer' must be boolean")

    # processing_history
    if "processing_history" not in entry:
        errors.append("missing 'processing_history' field")
    elif entry["processing_history"] is not None and not isinstance(entry["processing_history"], str):
        errors.append("'processing_history' must be string or null")

    # hierarchical_structure
    if "hierarchical_structure" not in entry:
        errors.append("missing 'hierarchical_structure' field")
    elif entry["hierarchical_structure"] is not None and not isinstance(entry["hierarchical_structure"], str):
        errors.append("'hierarchical_structure' must be string or null")

    # properties
    props = entry.get("properties")
    if not isinstance(props, dict):
        errors.append("missing or invalid 'properties' (must be dict)")
    else:
        prop_data = props.get(prop_key)
        if not isinstance(prop_data, dict):
            errors.append(f"properties missing '{prop_key}' (must be dict)")
        else:
            # value
            value = prop_data.get("value")
            if not isinstance(value, list) or len(value) == 0:
                errors.append(f"properties.{prop_key}.value must be a non-empty array")
            else:
                for i, v in enumerate(value):
                    if not isinstance(v, (int, float)):
                        errors.append(f"properties.{prop_key}.value[{i}] must be a number")

            # unit
            if "unit" not in prop_data:
                errors.append(f"properties.{prop_key} missing 'unit'")
            elif prop_data["unit"] is not None and not isinstance(prop_data["unit"], str):
                errors.append(f"properties.{prop_key}.unit must be string or null")

    is_valid = len(errors) == 0
    return is_valid, errors


def check_and_route(
    entries: list,
    prop_key: str,
    dataset_dir: str,
    filename: str | None = None,
) -> Tuple[int, int]:
    """
    对一组条目进行格式验证，合格的追加到 {prop_key}_extracted.json，
    不合格的追加到 {prop_key}_format_errors.json。

    存储格式为 JSON 数组，每次追加时读取已有数组、合并后重写。

    Args:
        entries: 数据条目列表
        prop_key: 目标属性键名
        dataset_dir: 数据集文件夹路径（如 "Tg/"）
        filename: 来源文件名（可选，附加到错误记录中方便定位）

    Returns:
        (valid_count, invalid_count)
    """
    os.makedirs(dataset_dir, exist_ok=True)

    valid_file = os.path.join(dataset_dir, f"{prop_key}_extracted.json")
    error_file = os.path.join(dataset_dir, f"{prop_key}_format_errors.json")

    valid_count = 0
    invalid_count = 0

    valid_entries = []
    error_entries = []

    for entry in entries:
        is_valid, errors = validate_entry(entry, prop_key)
        if is_valid:
            valid_entries.append(entry)
            valid_count += 1
        else:
            error_record = {
                "entry": entry,
                "errors": errors,
            }
            if filename:
                error_record["source_file"] = filename
            error_entries.append(error_record)
            invalid_count += 1

    # 追加合格条目
    if valid_entries:
        existing = []
        if os.path.exists(valid_file):
            try:
                with open(valid_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []
        existing.extend(valid_entries)
        with open(valid_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    # 追加不合格条目
    if error_entries:
        existing = []
        if os.path.exists(error_file):
            try:
                with open(error_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []
        existing.extend(error_entries)
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    return valid_count, invalid_count
