"""
PDF to Markdown conversion script (based on MinerU)
Converts PDF files from the pdf/ subdirectory of specified dataset folders to Markdown,
and saves them to the sibling md/ subdirectory.

Hardware: AMD (25 threads), 90GB RAM, RTX 5090 32GB, Linux
"""

import os

# Download models from modelscope instead of huggingface
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
# Reduce concurrency to 1 to avoid OOM
os.environ["MINERU_API_MAX_CONCURRENT_REQUESTS"] = "1"


import shutil
import subprocess
import glob
import time
from tqdm import tqdm

# ================= Configuration Parameters =================
# Dataset list to process, options: "Tg", "Tm", "n", "eps", "E", "UTS"
TARGET_DATASETS = ["Tg"]

# Number of PDFs per batch (batch processing avoids reloading the model, improving speed)
BATCH_SIZE = 50

# MinerU parsing backend: "hybrid-auto-engine" for GPU acceleration, "pipeline" for CPU only
BACKEND = "hybrid-auto-engine"

# Whether to enable formula parsing
FORMULA = True

# Whether to enable table parsing
TABLE = True

# Working root directory (directory of this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Temporary output directory name (use short name)
TEMP_DIR_NAME = "0"

# Batch ratio for small-model processing in hybrid-* backend
# Lower this value to reduce GPU memory per client; leave as None to skip
HYBRID_BATCH_RATIO = 1

# Force hybrid-* backend text extraction to use small model
# Default False; set to True to reduce hallucinations in some edge cases
HYBRID_FORCE_PIPELINE_ENABLE = True

# Number of threads for rendering PDF to images (Linux / macOS only)
# Default 4; increase to speed up rendering, decrease to reduce memory; leave as None to skip
PDF_RENDER_THREADS = None

# Single processing window size, affects memory usage and throughput for large documents
# Default 64, can be set to other positive integers; leave as None to skip
PROCESSING_WINDOW_SIZE = None

# PyTorch CUDA memory allocation strategy (Note: newer PyTorch renamed PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF)
# expandable_segments:True reduces GPU memory fragmentation, mitigates "free memory < requested" OOM
# Leave as None to use PyTorch default strategy
PYTORCH_ALLOC_CONF = "expandable_segments:True"

# vllm VLM engine GPU memory utilization cap (0.0-1.0)
# Default 0.5 (~15.7 GiB); lower this value to reduce VLM engine GPU memory preallocation
# Leave as None to use MinerU default value
VLM_GPU_MEMORY_UTILIZATION = 0.3
# =============================================

# Apply relevant environment variables
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


def clear_gpu_memory(wait: float = 5.0):
    """
    Terminate MinerU background fast_api service process to force GPU memory release.
    In hybrid-auto-engine mode, each mineru command starts a fast_api server in the background.
    If not terminated, this process continues occupying GPU memory between batches, causing OOM.
    """
    try:
        result = subprocess.run(
            ["pkill", "-f", "fast_api"],
            check=False, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  [GPU] MinerU fast_api process terminated, waiting {wait:.0f}s for GPU memory release...")
            time.sleep(wait)
        else:
            print("  [GPU] No residual fast_api process found.")
    except FileNotFoundError:
        pass  # Skip when pkill is unavailable (e.g., Windows)


def get_pending_files(pdf_dir: str, md_dir: str) -> list:
    """Compare pdf and md directories, return list of unconverted PDF files."""
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if os.path.exists(md_dir):
        done_stems = {os.path.splitext(f)[0] for f in os.listdir(md_dir) if f.lower().endswith(".md")}
    else:
        done_stems = set()

    pending = [f for f in pdf_files if os.path.splitext(f)[0] not in done_stems]
    return pending


def process_batch(batch_files: list, pdf_dir: str, md_dir: str, temp_dir: str):
    """
    Copy a batch of PDFs to a temp directory, call MinerU for batch processing,
    then migrate generated md files to the target md directory.
    """
    # Clean/create temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Create temp input and output directories
    temp_input = os.path.join(temp_dir, "input")
    temp_output = os.path.join(temp_dir, "output")
    os.makedirs(temp_input, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)

    # Copy PDFs to temp input directory (use short filename mapping to prevent path too long)
    name_map = {}  # short_name -> original_stem
    for i, pdf_name in enumerate(batch_files):
        short_name = f"{i}.pdf"
        src = os.path.join(pdf_dir, pdf_name)
        dst = os.path.join(temp_input, short_name)
        shutil.copy2(src, dst)
        name_map[str(i)] = os.path.splitext(pdf_name)[0]

    # Build MinerU command
    cmd = [
        "mineru",
        "-p", temp_input,
        "-o", temp_output,
        "-b", BACKEND,
        "-f", str(FORMULA).lower(),
        "-t", str(TABLE).lower(),
    ]

    try:
        subprocess.run(cmd, check=True, cwd=BASE_DIR)
    except subprocess.CalledProcessError as e:
        print(f"MinerU 处理失败: {e}")
        return

    # Find md files in output and migrate them
    os.makedirs(md_dir, exist_ok=True)

    # Use os.walk to collect all md files under temp_output, index by filename stem
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
            print(f"Warning: MD output not found for {original_stem}")
            print(f"  All md files in temp_output: {list(md_file_map.keys())}")
            continue

        md_dst = os.path.join(md_dir, f"{original_stem}.md")
        shutil.copy2(md_src, md_dst)

    # Clean temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    for dataset in TARGET_DATASETS:
        pdf_dir = os.path.join(BASE_DIR, dataset, "pdf")
        md_dir = os.path.join(BASE_DIR, dataset, "md")

        if not os.path.exists(pdf_dir):
            print(f"[{dataset}] pdf directory does not exist: {pdf_dir}, skipping.")
            continue

        pending = get_pending_files(pdf_dir, md_dir)
        total = len([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")])
        done = total - len(pending)

        print(f"\n[{dataset}] Total: {total} PDFs | Done: {done} | Pending: {len(pending)}")

        if not pending:
            print(f"[{dataset}] All files processed.")
            continue

        temp_dir = os.path.join(BASE_DIR, TEMP_DIR_NAME)

        # Process in batches of batch_size
        pbar = tqdm(total=len(pending), desc=f"[{dataset}] Conversion progress")
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]
            process_batch(batch, pdf_dir, md_dir, temp_dir)
            clear_gpu_memory()

            # Update progress bar (by actual success count)
            newly_done = get_pending_files(pdf_dir, md_dir)
            actual_done = len(pending) - len(newly_done)
            pbar.n = actual_done
            pbar.refresh()

        pbar.close()

        # Final statistics
        final_pending = get_pending_files(pdf_dir, md_dir)
        final_done = total - len(final_pending)
        print(f"[{dataset}] Processing complete. Success: {final_done}/{total}")


if __name__ == "__main__":
    main()
