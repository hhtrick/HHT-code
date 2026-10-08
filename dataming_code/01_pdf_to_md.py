"""
PDF to Markdown 转换脚本 (基于 MinerU)
将指定数据集文件夹下 pdf/ 子目录中的 PDF 文件转换为 Markdown，
并保存到同级 md/ 子目录中。

硬件环境: AMD (25线程), 90GB RAM, RTX 5090 32GB, Linux
"""

import os

# 指定从 modelscope 下载模型，而非 huggingface
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
# 降低并发数为1，避免OOM
os.environ["MINERU_API_MAX_CONCURRENT_REQUESTS"] = "1"


import shutil
import subprocess
import glob
import time
from tqdm import tqdm

# ================= 配置参数 =================
# 需要处理的数据集列表，可选: "Tg", "Tm", "n", "eps", "E", "UTS"
TARGET_DATASETS = ["Tg"]

# 每批次处理的 PDF 数量（批量处理避免反复加载模型，提升速度）
BATCH_SIZE = 50

# MinerU 解析后端，GPU 加速使用 "hybrid-auto-engine"，纯 CPU 使用 "pipeline"
BACKEND = "hybrid-auto-engine"

# 是否开启公式解析
FORMULA = True

# 是否开启表格解析
TABLE = True

# 工作根目录（当前脚本所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 临时输出目录名（使用短名称）
TEMP_DIR_NAME = "0"

# hybrid-* 后端中小模型处理的 batch 倍率
# 可通过降低该值来减少单个客户端的显存占用量，留空 None 则不设置
HYBRID_BATCH_RATIO = 1

# 是否强制 hybrid-* 后端中的文本提取部分使用小模型处理
# 默认 False；设为 True 可在某些极端情况下减少幻觉的发生
HYBRID_FORCE_PIPELINE_ENABLE = True

# 将 PDF 渲染为图片时使用的线程数（仅 Linux / macOS 生效）
# 默认为 4，可调大以加速渲染，调小以降低内存占用；留空 None 则不设置
PDF_RENDER_THREADS = None

# 单次处理窗口大小，影响大文档处理时的内存占用和吞吐表现
# 默认为 64，可设为其他正整数；留空 None 则不设置
PROCESSING_WINDOW_SIZE = None

# PyTorch CUDA 内存分配策略（注意：PyTorch 新版已将 PYTORCH_CUDA_ALLOC_CONF 更名为 PYTORCH_ALLOC_CONF）
# expandable_segments:True 可减少显存碎片，缓解 "free memory < requested" 类 OOM
# 留空 None 则不设置（使用 PyTorch 默认策略）
PYTORCH_ALLOC_CONF = "expandable_segments:True"

# vllm VLM 引擎 GPU 显存利用率上限（0.0-1.0）
# 默认 0.5（约 15.7 GiB），降低此值可减少 VLM 引擎的显存预分配量
# 留空 None 则使用 MinerU 默认值
VLM_GPU_MEMORY_UTILIZATION = 0.3
# =============================================

# 应用相关环境变量
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
    终止 MinerU 后台 fast_api 服务进程，强制释放其占用的 GPU 显存。
    hybrid-auto-engine 模式下每次调用 mineru 命令都会在后台启动 fast_api 服务器，
    若不主动终止，该进程会在批次间持续占用显存，导致下一批次因空闲显存不足而 OOM。
    """
    try:
        result = subprocess.run(
            ["pkill", "-f", "fast_api"],
            check=False, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  [GPU] 已终止 MinerU fast_api 进程，等待 {wait:.0f}s 释放显存...")
            time.sleep(wait)
        else:
            print("  [GPU] 未发现残留 fast_api 进程。")
    except FileNotFoundError:
        pass  # pkill 不可用时跳过（如 Windows）


def get_pending_files(pdf_dir: str, md_dir: str) -> list:
    """对比 pdf 目录和 md 目录，返回尚未转换的 PDF 文件列表。"""
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if os.path.exists(md_dir):
        done_stems = {os.path.splitext(f)[0] for f in os.listdir(md_dir) if f.lower().endswith(".md")}
    else:
        done_stems = set()

    pending = [f for f in pdf_files if os.path.splitext(f)[0] not in done_stems]
    return pending


def process_batch(batch_files: list, pdf_dir: str, md_dir: str, temp_dir: str):
    """
    将一批 PDF 复制到临时目录，调用 MinerU 批量处理，
    完成后将生成的 md 文件迁移到目标 md 目录。
    """
    # 清理/创建临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # 创建临时输入目录和输出目录
    temp_input = os.path.join(temp_dir, "input")
    temp_output = os.path.join(temp_dir, "output")
    os.makedirs(temp_input, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)

    # 复制 PDF 到临时输入目录（使用短文件名映射，防止路径过长）
    name_map = {}  # short_name -> original_stem
    for i, pdf_name in enumerate(batch_files):
        short_name = f"{i}.pdf"
        src = os.path.join(pdf_dir, pdf_name)
        dst = os.path.join(temp_input, short_name)
        shutil.copy2(src, dst)
        name_map[str(i)] = os.path.splitext(pdf_name)[0]

    # 构建 MinerU 命令
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

    # 从输出中找到 md 文件并迁移
    os.makedirs(md_dir, exist_ok=True)

    # 用 os.walk 收集 temp_output 下所有 md 文件，按文件名 stem 索引
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
            print(f"警告: 未找到 {original_stem} 对应的 MD 输出")
            print(f"  temp_output 内所有 md 文件: {list(md_file_map.keys())}")
            continue

        md_dst = os.path.join(md_dir, f"{original_stem}.md")
        shutil.copy2(md_src, md_dst)

    # 清理临时目录
    shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    for dataset in TARGET_DATASETS:
        pdf_dir = os.path.join(BASE_DIR, dataset, "pdf")
        md_dir = os.path.join(BASE_DIR, dataset, "md")

        if not os.path.exists(pdf_dir):
            print(f"[{dataset}] pdf 目录不存在: {pdf_dir}，跳过。")
            continue

        pending = get_pending_files(pdf_dir, md_dir)
        total = len([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")])
        done = total - len(pending)

        print(f"\n[{dataset}] 总计: {total} 个 PDF | 已完成: {done} | 待处理: {len(pending)}")

        if not pending:
            print(f"[{dataset}] 所有文件已处理完毕。")
            continue

        temp_dir = os.path.join(BASE_DIR, TEMP_DIR_NAME)

        # 按 batch_size 分批处理
        pbar = tqdm(total=len(pending), desc=f"[{dataset}] 转换进度")
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]
            process_batch(batch, pdf_dir, md_dir, temp_dir)
            clear_gpu_memory()

            # 更新进度条（按实际成功数更新）
            newly_done = get_pending_files(pdf_dir, md_dir)
            actual_done = len(pending) - len(newly_done)
            pbar.n = actual_done
            pbar.refresh()

        pbar.close()

        # 最终统计
        final_pending = get_pending_files(pdf_dir, md_dir)
        final_done = total - len(final_pending)
        print(f"[{dataset}] 处理完成。成功: {final_done}/{total}")


if __name__ == "__main__":
    main()
