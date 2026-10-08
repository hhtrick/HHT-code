"""
03_add_title.py — 为提取的 JSON 数据添加文献标题字段。

读取 {dataset}_extracted.json 中的每个条目，
从对应的 md 文件中提取第一个一级标题作为 title 字段，
插入到 doi 字段下方，保存为新文件 {dataset}_extracted_titled.json。
"""

import os
import re
import json

# ================= 配置参数 =================
# 需要处理的数据集列表，可选: "Tg", "Tm", "n", "eps", "E", "UTS"
TARGET_DATASETS = ["E"]

# 工作根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =============================================


def extract_title_from_md(md_path: str) -> str | None:
    """从 Markdown 文件中提取第一个一级标题（# 开头的行）。"""
    if not os.path.exists(md_path):
        return None
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 匹配一级标题：以 # 开头，后跟空格，且不是 ## 及更多
            if re.match(r"^#\s+", line) and not line.startswith("##"):
                # 去掉开头的 # 和空格
                return line.lstrip("#").strip()
    return None


def add_title_to_entry(entry: dict, title: str | None) -> dict:
    """在 doi 字段下方插入 title 字段，返回新的有序字典。"""
    new_entry = {}
    title_inserted = False
    for k, v in entry.items():
        new_entry[k] = v
        if k == "doi" and not title_inserted:
            new_entry["title"] = title
            title_inserted = True
    # 如果 entry 中没有 doi 字段，也要确保 title 被添加
    if not title_inserted:
        new_entry["title"] = title
    return new_entry


def main():
    for dataset in TARGET_DATASETS:
        dataset_dir = os.path.join(BASE_DIR, dataset)
        md_dir = os.path.join(dataset_dir, "md")
        input_file = os.path.join(dataset_dir, f"{dataset}_extracted.json")
        output_file = os.path.join(dataset_dir, f"{dataset}_extracted_titled.json")

        if not os.path.exists(input_file):
            print(f"[{dataset}] 提取文件不存在: {input_file}，跳过。")
            continue

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        titled_count = 0
        missing_count = 0
        result = []

        for entry in data:
            filename = entry.get("filename", "")
            md_path = os.path.join(md_dir, f"{filename}.md")
            title = extract_title_from_md(md_path)

            if title:
                titled_count += 1
            else:
                missing_count += 1
                print(f"  [WARN] 未找到标题: {filename}")

            new_entry = add_title_to_entry(entry, title)
            result.append(new_entry)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[{dataset}] 完成。共 {len(data)} 条 | 成功提取标题: {titled_count} | 未找到标题: {missing_count}")
        print(f"  输出文件: {output_file}")


if __name__ == "__main__":
    main()
