"""Extract polymer records from Markdown, review them with a second LLM, and validate the schema."""

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


CONCURRENCY = 24


MAX_RETRIES = 3


TARGET_DATASETS = ["E"]

# "test": sample three pending articles and save reasoning; "prod": process all pending articles.
MODE = "prod"

# Reprocess all sources, including completed ones; existing outputs are appended.
IGNORE_PROGRESS = False


MODEL_STAGE1 = "qwen3.5-plus-2026-02-15"


MODEL_STAGE2 = "glm-5"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=httpx.Timeout(timeout=600.0, connect=10.0),
)


def _progress_file(dataset: str) -> str:
    return os.path.join(BASE_DIR, dataset, f"{dataset}_extraction_progress.txt")


def load_done_files(dataset: str) -> set:
    """Return completed filenames, or an empty set when IGNORE_PROGRESS is enabled."""
    if IGNORE_PROGRESS:
        return set()
    pf = _progress_file(dataset)
    if not os.path.exists(pf):
        return set()
    with open(pf, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(dataset: str, filename: str):
    """Record a completed source filename for resuming the pipeline."""
    pf = _progress_file(dataset)
    with open(pf, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


def _no_data_file(dataset: str) -> str:
    return os.path.join(BASE_DIR, dataset, f"{dataset}_no_data.txt")


def log_no_data(dataset: str, filename: str):
    """Record a source filename for which extraction returned no records."""
    ndf = _no_data_file(dataset)
    with open(ndf, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


async def call_llm(
    prompt: str,
    model: str,
    sem: asyncio.Semaphore,
) -> dict:
    """Call the LLM with retries and return content plus any reasoning output."""
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
                    f"{2 ** attempt}s before retry...",
                    flush=True,
                )
                await asyncio.sleep(2 ** attempt)
            else:
                print(
                    f"  [ERROR] {model} failed ({MAX_RETRIES} attempts): "
                    f"[{exc_type}] {e}",
                    flush=True,
                )
                print(
                    f"  [ERROR] Last attempt took {elapsed:.2f}s; full traceback:\n"
                    + traceback.format_exc(),
                    flush=True,
                )
                return {"content": "", "reasoning": ""}

    return {"content": "", "reasoning": ""}


def _parse_json_array(text: str) -> list | None:
    """Parse a JSON array, accepting a single object as a one-element array."""
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

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None


async def process_single_file(
    md_path: str,
    prop_key: str,
    dataset_dir: str,
    sem1: asyncio.Semaphore,
    sem2: asyncio.Semaphore,
) -> dict:
    """Extract and review one article using separate concurrency limits for the two stages."""
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


    prompt1 = get_extraction_prompt(prop_key, md_text)
    resp1 = await call_llm(prompt1, MODEL_STAGE1, sem1)

    stage1_data = _parse_json_array(resp1["content"])
    if stage1_data is None:
        result["error"] = f"Stage 1 output could not be parsed as JSON: {resp1['content'][:200]}"
        result["stage1_reasoning"] = resp1["reasoning"]
        return result

    result["stage1_entries"] = stage1_data
    result["stage1_reasoning"] = resp1["reasoning"]


    extracted_json_str = json.dumps(stage1_data, ensure_ascii=False, indent=2)
    prompt2 = get_review_prompt(prop_key, extracted_json_str, md_text)
    resp2 = await call_llm(prompt2, MODEL_STAGE2, sem2)

    stage2_data = _parse_json_array(resp2["content"])
    result["stage2_reasoning"] = resp2["reasoning"]

    if stage2_data is None:
        result["error"] = "Stage 2 review failed or returned invalid JSON; extraction was not accepted."
        return result
    if len(stage2_data) > 0:

        result["stage2_entries"] = stage2_data
        result["final_entries"] = stage2_data
    else:

        result["final_entries"] = stage1_data

    return result


async def main():
    # Separate limits prevent extraction requests from starving the review stage.
    sem1 = asyncio.Semaphore(CONCURRENCY)
    sem2 = asyncio.Semaphore(CONCURRENCY)


    print(f"[INFO] Python platform: {platform.system()} {platform.version()}", flush=True)
    print(f"[INFO] Mode: {MODE} | Concurrency: {CONCURRENCY} | Max retries: {MAX_RETRIES}", flush=True)
    print(f"[INFO] Extraction model: {MODEL_STAGE1} | Review model: {MODEL_STAGE2}", flush=True)

    for dataset in TARGET_DATASETS:
        md_dir = os.path.join(BASE_DIR, dataset, "md")
        dataset_dir = os.path.join(BASE_DIR, dataset)

        if not os.path.exists(md_dir):
            print(f"[{dataset}] Markdown directory does not exist: {md_dir}; skipping.")
            continue


        all_md_files = sorted([
            f for f in os.listdir(md_dir) if f.lower().endswith(".md")
        ])

        if not all_md_files:
            print(f"[{dataset}] Markdown directory is empty; skipping.")
            continue


        done_files = load_done_files(dataset)
        pending_files = [f for f in all_md_files if os.path.splitext(f)[0] not in done_files]


        if MODE == "test":
            pending_files = random.sample(pending_files, min(3, len(pending_files)))

        print(f"\n[{dataset}] Total: {len(all_md_files)} | Completed: {len(done_files)} | Pending this run: {len(pending_files)} | Mode: {MODE}")

        if not pending_files:
            print(f"[{dataset}] No pending files.")
            continue


        test_output_file = os.path.join(dataset_dir, f"{dataset}_test_output.json")


        async def _worker(md_filename):
            md_path = os.path.join(md_dir, md_filename)
            return await process_single_file(md_path, dataset, dataset_dir, sem1, sem2)

        tasks = [_worker(f) for f in pending_files]

        total_valid = 0
        total_invalid = 0
        total_errors = 0
        total_no_data = 0

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"[{dataset}] Extracting"):
            result = await coro
            filename = result["filename"]

            if result["error"]:
                print(f"  [WARN] {filename}: {result['error']}")
                total_errors += 1
                # Leave failed sources pending so a later run can retry them.
                continue

            final_entries = result["final_entries"] or []


            if not final_entries:
                total_no_data += 1
                log_no_data(dataset, filename)

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


            for entry in final_entries:
                if isinstance(entry, dict):

                    new_entry = {}
                    for k, v in entry.items():
                        new_entry[k] = v
                        if k == "doi":
                            new_entry["filename"] = filename
                    if "filename" not in new_entry:
                        new_entry["filename"] = filename
                    entry.clear()
                    entry.update(new_entry)


            v, iv = check_and_route(final_entries, dataset, dataset_dir, filename)
            total_valid += v
            total_invalid += iv


            if MODE == "test":
                test_record = {
                    "filename": filename,
                    "stage1_entries": result["stage1_entries"],
                    "stage1_reasoning": result["stage1_reasoning"],
                    "stage2_entries": result["stage2_entries"],
                    "stage2_reasoning": result["stage2_reasoning"],
                    "final_entries": final_entries,
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

        print(f"[{dataset}] Finished. Accepted: {total_valid} | Rejected: {total_invalid} | No data: {total_no_data} | Errors: {total_errors}")


if __name__ == "__main__":
    # Set the Windows policy before creating the event loop.
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
