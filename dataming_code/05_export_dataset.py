"""
05_export_dataset.py — 导出最终数据集。

读取 04_data_processor.py 输出在 final/ 文件夹下的 JSON 文件，
去掉每个条目的 "filename" 字段，保存到 dataset/ 文件夹下。
"""

import os
import json

# ================= 配置参数 =================
# 需要处理的数据集列表，可选: "Tg", "Tm", "n", "eps", "E", "UTS"
TARGET_DATASETS = ["E"]

# 工作根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =============================================


def main():
    final_dir = os.path.join(BASE_DIR, "final")
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    for dataset in TARGET_DATASETS:
        input_file = os.path.join(final_dir, f"{dataset}.json")
        output_file = os.path.join(dataset_dir, f"{dataset}.json")

        if not os.path.exists(input_file):
            print(f"[{dataset}] 输入文件不存在: {input_file}，跳过。")
            continue

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = []
        for entry in data:
            if isinstance(entry, dict):
                new_entry = {k: v for k, v in entry.items() if k != "filename"}
                result.append(new_entry)
            else:
                result.append(entry)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[{dataset}] 完成。共 {len(result)} 条，已去除 filename 字段。")
        print(f"  输出文件: {output_file}")


if __name__ == "__main__":
    main()
