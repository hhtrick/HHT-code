"""Convert property-specific PDFs to Markdown using the MinerU CLI."""

import os

# Download MinerU models from ModelScope.
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
# Limit concurrent MinerU clients to reduce GPU memory usage.
os.environ["MINERU_API_MAX_CONCURRENT_REQUESTS"] = "1"


import shutil
import subprocess
import signal
import time
from tqdm import tqdm


TARGET_DATASETS = ["Tg"]

# PDFs per batch; batching avoids repeated model loading.
BATCH_SIZE = 50

# MinerU backend: "hybrid-auto-engine" for GPU or "pipeline" for CPU.
BACKEND = "hybrid-auto-engine"

# Enable formula parsing.
FORMULA = True

# Enable table parsing.
TABLE = True


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


TEMP_DIR_NAME = "0"

# Batch multiplier for the hybrid backend; reduce it to lower memory usage.
HYBRID_BATCH_RATIO = 1

# Force the hybrid backend to use its smaller text-extraction model.
HYBRID_FORCE_PIPELINE_ENABLE = True

# PDF rendering threads (Linux/macOS only).
PDF_RENDER_THREADS = None

# Processing window size; smaller values reduce peak memory usage.
PROCESSING_WINDOW_SIZE = None

# Optional PyTorch allocator settings.
PYTORCH_ALLOC_CONF = "expandable_segments:True"

# VLM GPU memory utilization limit (0.0-1.0); None uses the MinerU default.
VLM_GPU_MEMORY_UTILIZATION = 0.3


if HYBRID_BATCH_RATIO is not None:
    os.environ["MINERU_HYBRID_BATCH_RATIO"] = str(HYBRID_BATCH_RATIO)
os.environ["MINERU_HYBRID_FORCE_PIPELINE_ENABLE"] = str(HYBRID_FORCE_PIPELINE_ENABLE).lower()
if PDF_RENDER_THREADS is not None:
    os.environ["MINERU_PDF_RENDER_THREADS"] = str(PDF_RENDER_THREADS)
if PROCESSING_WINDOW_SIZE is not None:
    os.environ["MINERU_PROCESSING_WINDOW_SIZE"] = str(PROCESSING_WINDOW_SIZE)
if PYTORCH_ALLOC_CONF is not None:
    os.environ["PYTORCH_ALLOC_CONF"] = PYTORCH_ALLOC_CONF
if VLM_GPU_MEMORY_UTILIZATION is not None:
    os.environ["MINERU_VLM_GPU_MEMORY_UTILIZATION"] = str(VLM_GPU_MEMORY_UTILIZATION)


def stop_batch_workers(process_group: int, wait: float = 5.0):
    """Release workers in this batch's process group without stopping other jobs."""
    if os.name != "posix":
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(wait)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def get_pending_files(pdf_dir: str, md_dir: str) -> list:
    """Return PDF filenames without a corresponding Markdown output."""
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if os.path.exists(md_dir):
        done_stems = {os.path.splitext(f)[0] for f in os.listdir(md_dir) if f.lower().endswith(".md")}
    else:
        done_stems = set()

    pending = [f for f in pdf_files if os.path.splitext(f)[0] not in done_stems]
    return pending


def process_batch(batch_files: list, pdf_dir: str, md_dir: str, temp_dir: str):
    """Parse a batch of PDFs and copy Markdown outputs to the dataset directory."""

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)


    temp_input = os.path.join(temp_dir, "input")
    temp_output = os.path.join(temp_dir, "output")
    os.makedirs(temp_input, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)


    name_map = {}  # short_name -> original_stem
    for i, pdf_name in enumerate(batch_files):
        short_name = f"{i}.pdf"
        src = os.path.join(pdf_dir, pdf_name)
        dst = os.path.join(temp_input, short_name)
        shutil.copy2(src, dst)
        name_map[str(i)] = os.path.splitext(pdf_name)[0]


    cmd = [
        "mineru",
        "-p", temp_input,
        "-o", temp_output,
        "-b", BACKEND,
        "-f", str(FORMULA).lower(),
        "-t", str(TABLE).lower(),
    ]

    process = subprocess.Popen(cmd, cwd=BASE_DIR, start_new_session=(os.name == "posix"))
    try:
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, cmd)
    except subprocess.CalledProcessError as e:
        print(f"MinerU failed: {e}")
        return
    finally:
        stop_batch_workers(process.pid)


    os.makedirs(md_dir, exist_ok=True)


    md_file_map = {}  # stem -> full_path
    for root, dirs, files in os.walk(temp_output):
        for f in files:
            if f.lower().endswith(".md"):
                stem = os.path.splitext(f)[0]
                md_file_map[stem] = os.path.join(root, f)

    for short_stem, original_stem in name_map.items():
        if short_stem in md_file_map:
            md_src = md_file_map[short_stem]
        else:
            print(f"Warning: no Markdown output for {original_stem}")
            print(f"  Markdown files in temporary output: {list(md_file_map.keys())}")
            continue

        md_dst = os.path.join(md_dir, f"{original_stem}.md")
        shutil.copy2(md_src, md_dst)


    shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    for dataset in TARGET_DATASETS:
        pdf_dir = os.path.join(BASE_DIR, dataset, "pdf")
        md_dir = os.path.join(BASE_DIR, dataset, "md")

        if not os.path.exists(pdf_dir):
            print(f"[{dataset}] PDF directory does not exist: {pdf_dir}; skipping.")
            continue

        pending = get_pending_files(pdf_dir, md_dir)
        total = len([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")])
        done = total - len(pending)

        print(f"\n[{dataset}] Total: {total} PDFs | Completed: {done} | Pending: {len(pending)}")

        if not pending:
            print(f"[{dataset}] All files have been processed.")
            continue

        temp_dir = os.path.join(BASE_DIR, TEMP_DIR_NAME)


        pbar = tqdm(total=len(pending), desc=f"[{dataset}] Converting")
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]
            process_batch(batch, pdf_dir, md_dir, temp_dir)


            newly_done = get_pending_files(pdf_dir, md_dir)
            actual_done = len(pending) - len(newly_done)
            pbar.n = actual_done
            pbar.refresh()

        pbar.close()


        final_pending = get_pending_files(pdf_dir, md_dir)
        final_done = total - len(final_pending)
        print(f"[{dataset}] Finished. Successful: {final_done}/{total}")


if __name__ == "__main__":
    main()
