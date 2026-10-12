"""Export generated records from final/ to dataset/, removing the temporary filename field."""

import os
import json


TARGET_DATASETS = ["E"]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    final_dir = os.path.join(BASE_DIR, "final")
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    for dataset in TARGET_DATASETS:
        input_file = os.path.join(final_dir, f"{dataset}.json")
        output_file = os.path.join(dataset_dir, f"{dataset}.json")

        if not os.path.exists(input_file):
            print(f"[{dataset}] Input file does not exist: {input_file}; skipping.")
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

        print(f"[{dataset}] Finished. Records: {len(result)}; removed the filename field.")
        print(f"  Output file: {output_file}")


if __name__ == "__main__":
    main()
