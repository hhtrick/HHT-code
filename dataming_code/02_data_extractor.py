"""
data_extractor.py — Two-stage LLM data extraction main pipeline.

Stage 1: qwen3.5-plus-2026-02-15 extracts structured JSON data from Markdown
Stage 2: glm-5 reviews and corrects Stage 1 output
Final: format_checker validates format and routes to appropriate files
"""

import os
import json
import asyncio
import random
import platform
import httpx
import traceback
import time
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

from prompt import get_extraction_prompt, get_review_prompt
from format_checker import check_and_route

load_dotenv()

# ================= Configuration Parameters =================
# Async concurrency count
CONCURRENCY = 24

# Max retry count on exceptions
MAX_RETRIES = 3

# Target dataset list, options: "Tg", "Tm", "n", "eps", "E", "UTS"
TARGET_DATASETS = ["E"]

# Processing mode: "test" — random 3 files + save reasoning chains; "prod" — full processing
MODE = "prod"

# Ignore progress file, force reprocess all files (use with caution when previous run failed entirely)
# Note: setting to True will reprocess ALL files including previously successful ones
IGNORE_PROGRESS = False

# Stage 1 model
MODEL_STAGE1 = "qwen3.5-plus-2026-02-15"

# Stage 2 model
MODEL_STAGE2 = "glm-5"

# Working root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =============================================

client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=httpx.Timeout(timeout=600.0, connect=10.0),
)


# ===========================================================================
# Progress Management (Checkpoint Resume)
# ===========================================================================
def _progress_file(dataset: str) -> str:
    return os.path.join(BASE_DIR, dataset, f"{dataset}_extraction_progress.txt")


def load_done_files(dataset: str) -> set:
    """Load set of completed filenames. Returns empty set if IGNORE_PROGRESS=True."""
    if IGNORE_PROGRESS:
        return set()
    pf = _progress_file(dataset)
    if not os.path.exists(pf):
        return set()
    with open(pf, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(dataset: str, filename: str):
    """Mark a file as processed."""
    pf = _progress_file(dataset)
    with open(pf, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


def _no_data_file(dataset: str) -> str:
    return os.path.join(BASE_DIR, dataset, f"{dataset}_no_data.txt")


def log_no_data(dataset: str, filename: str):
    """Log filename that failed to extract data."""
    ndf = _no_data_file(dataset)
    with open(ndf, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


# ===========================================================================
# LLM Calling
# ===========================================================================
async def call_llm(
    prompt: str,
    model: str,
    sem: asyncio.Semaphore,
) -> dict:
    """
    Call LLM (with thinking mode enabled), returns {"content": ..., "reasoning": ...}.
    Includes retry logic with detailed diagnostic logging.
    """
    for attempt in range(MAX_RETRIES):
        t_start = time.monotonic()
        try:
            async with sem:
                response = await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": True},
                )

            content = response.choices[0].message.content or ""
            msg_dict = response.choices[0].message.model_dump()
            reasoning = msg_dict.get("reasoning_content", "")

            # Clean possible markdown wrapping
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            return {"content": content, "reasoning": reasoning}

        except Exception as e:
            elapsed = time.monotonic() - t_start
            exc_type = type(e).__name__
            if attempt < MAX_RETRIES - 1:
                print(
                    f"  [WARN] {model} attempt {attempt + 1} failed "
                    f"(elapsed {elapsed:.2f}s): [{exc_type}] {e}, "
                    f"retrying in {2 ** attempt}s...",
                    flush=True,
                )
                await asyncio.sleep(2 ** attempt)
            else:
                print(
                    f"  [ERROR] {model} call failed ({MAX_RETRIES} attempts): "
                    f"[{exc_type}] {e}",
                    flush=True,
                )
                print(
                    f"  [ERROR] Last attempt took {elapsed:.2f}s, full traceback:\n"
                    + traceback.format_exc(),
                    flush=True,
                )
                return {"content": "", "reasoning": ""}

    return {"content": "", "reasoning": ""}


def _parse_json_array(text: str) -> list | None:
    """Try to parse a JSON array from text."""
    text = text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return None
    except json.JSONDecodeError:
        # Try to find first [ and last ]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None


# ===========================================================================
# 单文件处理流程
# ===========================================================================
async def process_single_file(
    md_path: str,
    prop_key: str,
    dataset_dir: str,
    sem1: asyncio.Semaphore,
    sem2: asyncio.Semaphore,
) -> dict:
    """
    Process a single Markdown file with two-stage extraction.
    sem1: dedicated to Stage 1, sem2: dedicated to Stage 2, independent to avoid mutual blocking.

    Returns:
        Result dict containing stage1/stage2 data and filename metadata.
    """
    filename = os.path.splitext(os.path.basename(md_path))[0]

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    result = {
        "filename": filename,
        "stage1_entries": None,
        "stage2_entries": None,
        "final_entries": None,
        "stage1_reasoning": None,
        "stage2_reasoning": None,
        "error": None,
    }

    # --- Stage 1: Extraction ---
    prompt1 = get_extraction_prompt(prop_key, md_text)
    resp1 = await call_llm(prompt1, MODEL_STAGE1, sem1)

    stage1_data = _parse_json_array(resp1["content"])
    if stage1_data is None:
        result["error"] = f"Stage 1 returned unparseable JSON: {resp1['content'][:200]}"
        result["stage1_reasoning"] = resp1["reasoning"]
        return result

    result["stage1_entries"] = stage1_data
    result["stage1_reasoning"] = resp1["reasoning"]

    # --- Stage 2: Review ---
    extracted_json_str = json.dumps(stage1_data, ensure_ascii=False, indent=2)
    prompt2 = get_review_prompt(prop_key, extracted_json_str, md_text)
    resp2 = await call_llm(prompt2, MODEL_STAGE2, sem2)

    stage2_data = _parse_json_array(resp2["content"])
    result["stage2_reasoning"] = resp2["reasoning"]

    if stage2_data is not None and len(stage2_data) > 0:
        # Review found issues, use corrected data
        result["stage2_entries"] = stage2_data
        result["final_entries"] = stage2_data
    else:
        # Review passed, use Stage 1 data
        result["final_entries"] = stage1_data

    return result


# ===========================================================================
# 主流程
# ===========================================================================
async def main():
    # Two stages use independent semaphores to prevent Stage 1 coroutines from starving Stage 2
    sem1 = asyncio.Semaphore(CONCURRENCY)  # 阶段一专用
    sem2 = asyncio.Semaphore(CONCURRENCY)  # 阶段二专用

    # --- Environment Info ---
    print(f"[INFO] Python platform: {platform.system()} {platform.version()}", flush=True)
    print(f"[INFO] Mode: {MODE} | Concurrency: {CONCURRENCY} | Max retries: {MAX_RETRIES}", flush=True)
    print(f"[INFO] Stage 1 model: {MODEL_STAGE1} | Stage 2 model: {MODEL_STAGE2}", flush=True)

    for dataset in TARGET_DATASETS:
        md_dir = os.path.join(BASE_DIR, dataset, "md")
        dataset_dir = os.path.join(BASE_DIR, dataset)

        if not os.path.exists(md_dir):
            print(f"[{dataset}] md directory does not exist: {md_dir}, skipping.")
            continue

        # Collect all md files
        all_md_files = sorted([
            f for f in os.listdir(md_dir) if f.lower().endswith(".md")
        ])

        if not all_md_files:
            print(f"[{dataset}] md directory is empty, skipping.")
            continue

        # Checkpoint resume: filter out already processed files
        done_files = load_done_files(dataset)
        pending_files = [f for f in all_md_files if os.path.splitext(f)[0] not in done_files]

        # Test mode: randomly sample 3
        if MODE == "test":
            pending_files = random.sample(pending_files, min(3, len(pending_files)))

        print(f"\n[{dataset}] Total: {len(all_md_files)} | Done: {len(done_files)} | Processing: {len(pending_files)} | Mode: {MODE}")

        if not pending_files:
            print(f"[{dataset}] No files to process.")
            continue

        # Test mode output file (includes reasoning chains)
        test_output_file = os.path.join(dataset_dir, f"{dataset}_test_output.json")

        # Process files one by one (Stage 1/Stage 2 each controlled by independent semaphores)
        async def _worker(md_filename):
            md_path = os.path.join(md_dir, md_filename)
            return await process_single_file(md_path, dataset, dataset_dir, sem1, sem2)

        tasks = [_worker(f) for f in pending_files]

        total_valid = 0
        total_invalid = 0
        total_errors = 0
        total_no_data = 0

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"[{dataset}] Extraction progress"):
            result = await coro
            filename = result["filename"]

            if result["error"]:
                print(f"  [WARN] {filename}: {result['error']}")
                total_errors += 1
                # Even on error, mark as processed to avoid getting stuck repeatedly
                mark_done(dataset, filename)
                continue

            final_entries = result["final_entries"] or []

            # No data extracted: log filename, skip further processing
            if not final_entries:
                total_no_data += 1
                log_no_data(dataset, filename)
                # Test mode still saves reasoning chains for debugging
                if MODE == "test":
                    test_record = {
                        "filename": filename,
                        "stage1_entries": result["stage1_entries"],
                        "stage1_reasoning": result["stage1_reasoning"],
                        "stage2_entries": result["stage2_entries"],
                        "stage2_reasoning": result["stage2_reasoning"],
                        "final_entries": [],
                        "no_data": True,
                    }
                    existing = []
                    if os.path.exists(test_output_file):
                        try:
                            with open(test_output_file, "r", encoding="utf-8") as f:
                                existing = json.load(f)
                        except (json.JSONDecodeError, IOError):
                            existing = []
                    existing.append(test_record)
                    with open(test_output_file, "w", encoding="utf-8") as f:
                        json.dump(existing, f, ensure_ascii=False, indent=2)
                mark_done(dataset, filename)
                continue

            # Add filename field to each entry (below doi)
            for entry in final_entries:
                if isinstance(entry, dict):
                    # Rebuild dict to ensure filename appears after doi
                    new_entry = {}
                    for k, v in entry.items():
                        new_entry[k] = v
                        if k == "doi":
                            new_entry["filename"] = filename
                    if "filename" not in new_entry:
                        new_entry["filename"] = filename
                    entry.clear()
                    entry.update(new_entry)

            # Format check and routing
            v, iv = check_and_route(final_entries, dataset, dataset_dir, filename)
            total_valid += v
            total_invalid += iv

            # Test mode: save reasoning chains
            if MODE == "test":
                test_record = {
                    "filename": filename,
                    "stage1_entries": result["stage1_entries"],
                    "stage1_reasoning": result["stage1_reasoning"],
                    "stage2_entries": result["stage2_entries"],
                    "stage2_reasoning": result["stage2_reasoning"],
                    "final_entries": final_entries,
                }
                # Append to file
                existing = []
                if os.path.exists(test_output_file):
                    try:
                        with open(test_output_file, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        existing = []
                existing.append(test_record)
                with open(test_output_file, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)

            # Mark as done
            mark_done(dataset, filename)

        print(f"[{dataset}] Done. Valid: {total_valid} | Invalid: {total_invalid} | No data: {total_no_data} | Errors: {total_errors}")


if __name__ == "__main__":
    # Must set before asyncio.run(), otherwise it won't affect current loop
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
