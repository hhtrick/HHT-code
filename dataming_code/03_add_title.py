"""Add the first Markdown level-one heading as the title of each extracted record."""

import os
import re
import json


TARGET_DATASETS = ["E"]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_title_from_md(md_path: str) -> str | None:
    """Read the first level-one Markdown heading as the article title."""
    if not os.path.exists(md_path):
        return None
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if re.match(r"^#\s+", line) and not line.startswith("##"):

                return line.lstrip("#").strip()
    return None


def add_title_to_entry(entry: dict, title: str | None) -> dict:
    """Insert the title immediately after DOI, or append it when DOI is absent."""
    new_entry = {}
    title_inserted = False
    for k, v in entry.items():
        new_entry[k] = v
        if k == "doi" and not title_inserted:
            new_entry["title"] = title
            title_inserted = True

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
            print(f"[{dataset}] Extraction file does not exist: {input_file}; skipping.")
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
                print(f"  [WARN] Title not found: {filename}")

            new_entry = add_title_to_entry(entry, title)
            result.append(new_entry)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[{dataset}] Finished. Records: {len(data)} | Titles found: {titled_count} | Title not found: {missing_count}")
        print(f"  Output file: {output_file}")


if __name__ == "__main__":
    main()
