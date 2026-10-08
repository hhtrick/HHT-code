"""
data_extractor.py — 两阶段 LLM 数据提取主流程。

阶段一: qwen3.5-plus-2026-02-15 从 Markdown 中提取结构化 JSON 数据
阶段二: glm-5 审查并修正阶段一的输出
最后: format_checker 验证格式并分流保存
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

# ================= 配置参数 =================
# 异步并发数
CONCURRENCY = 24

# 异常最大重试次数
MAX_RETRIES = 3

# 目标数据集列表，可选: "Tg", "Tm", "n", "eps", "E", "UTS"
TARGET_DATASETS = ["E"]

# 处理模式: "test" — 随机 3 个文件 + 保存思维链; "prod" — 全量处理
MODE = "prod"

# 忽略进度文件，强制重新处理所有文件（用于在上一次全部失败后重新运行）
# 注意：设为 True 会重新处理包括已成功文件在内的全部文件，请谨慎使用
IGNORE_PROGRESS = False

# 阶段一模型
MODEL_STAGE1 = "qwen3.5-plus-2026-02-15"

# 阶段二模型
MODEL_STAGE2 = "glm-5"

# 工作根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =============================================

client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=httpx.Timeout(timeout=600.0, connect=10.0),
)


# ===========================================================================
# 进度管理（断电恢复）
# ===========================================================================
def _progress_file(dataset: str) -> str:
    return os.path.join(BASE_DIR, dataset, f"{dataset}_extraction_progress.txt")


def load_done_files(dataset: str) -> set:
    """加载已完成文件名集合。若 IGNORE_PROGRESS=True 则返回空集合。"""
    if IGNORE_PROGRESS:
        return set()
    pf = _progress_file(dataset)
    if not os.path.exists(pf):
        return set()
    with open(pf, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(dataset: str, filename: str):
    """标记一个文件为已处理。"""
    pf = _progress_file(dataset)
    with open(pf, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


def _no_data_file(dataset: str) -> str:
    return os.path.join(BASE_DIR, dataset, f"{dataset}_no_data.txt")


def log_no_data(dataset: str, filename: str):
    """记录未能提取到数据的文件名。"""
    ndf = _no_data_file(dataset)
    with open(ndf, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


# ===========================================================================
# LLM 调用
# ===========================================================================
async def call_llm(
    prompt: str,
    model: str,
    sem: asyncio.Semaphore,
) -> dict:
    """
    调用 LLM（带思考模式），返回 {"content": ..., "reasoning": ...}。
    含重试逻辑，附带详细诊断日志。
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

            # 清理可能的 markdown 包裹
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
                    f"  [WARN] {model} 第 {attempt + 1} 次尝试失败 "
                    f"(耗时 {elapsed:.2f}s): [{exc_type}] {e}，"
                    f"{2 ** attempt}s 后重试...",
                    flush=True,
                )
                await asyncio.sleep(2 ** attempt)
            else:
                print(
                    f"  [ERROR] {model} 调用失败 ({MAX_RETRIES} 次): "
                    f"[{exc_type}] {e}",
                    flush=True,
                )
                print(
                    f"  [ERROR] 最后一次尝试耗时 {elapsed:.2f}s，完整堆栈:\n"
                    + traceback.format_exc(),
                    flush=True,
                )
                return {"content": "", "reasoning": ""}

    return {"content": "", "reasoning": ""}


def _parse_json_array(text: str) -> list | None:
    """尝试从文本中解析出 JSON 数组。"""
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
        # 尝试找到第一个 [ 和最后一个 ]
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
    对单个 Markdown 文件执行两阶段提取。
    sem1 专用于阶段一，sem2 专用于阶段二，互相独立，避免相互阻塞。

    Returns:
        结果字典，包含 stage1/stage2 数据以及文件名等元信息。
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

    # --- 阶段一: 提取 ---
    prompt1 = get_extraction_prompt(prop_key, md_text)
    resp1 = await call_llm(prompt1, MODEL_STAGE1, sem1)

    stage1_data = _parse_json_array(resp1["content"])
    if stage1_data is None:
        result["error"] = f"Stage 1 返回无法解析为 JSON: {resp1['content'][:200]}"
        result["stage1_reasoning"] = resp1["reasoning"]
        return result

    result["stage1_entries"] = stage1_data
    result["stage1_reasoning"] = resp1["reasoning"]

    # --- 阶段二: 审查 ---
    extracted_json_str = json.dumps(stage1_data, ensure_ascii=False, indent=2)
    prompt2 = get_review_prompt(prop_key, extracted_json_str, md_text)
    resp2 = await call_llm(prompt2, MODEL_STAGE2, sem2)

    stage2_data = _parse_json_array(resp2["content"])
    result["stage2_reasoning"] = resp2["reasoning"]

    if stage2_data is not None and len(stage2_data) > 0:
        # 审查发现问题，使用修正后的数据
        result["stage2_entries"] = stage2_data
        result["final_entries"] = stage2_data
    else:
        # 审查通过，使用阶段一数据
        result["final_entries"] = stage1_data

    return result


# ===========================================================================
# 主流程
# ===========================================================================
async def main():
    # 两个阶段使用独立信号量，避免 stage-1 协程长期压占信号量导致 stage-2 饥饿
    sem1 = asyncio.Semaphore(CONCURRENCY)  # 阶段一专用
    sem2 = asyncio.Semaphore(CONCURRENCY)  # 阶段二专用

    # --- 环境信息 ---
    print(f"[INFO] Python 平台: {platform.system()} {platform.version()}", flush=True)
    print(f"[INFO] 模式: {MODE} | 并发数: {CONCURRENCY} | 最大重试: {MAX_RETRIES}", flush=True)
    print(f"[INFO] 阶段一模型: {MODEL_STAGE1} | 阶段二模型: {MODEL_STAGE2}", flush=True)

    for dataset in TARGET_DATASETS:
        md_dir = os.path.join(BASE_DIR, dataset, "md")
        dataset_dir = os.path.join(BASE_DIR, dataset)

        if not os.path.exists(md_dir):
            print(f"[{dataset}] md 目录不存在: {md_dir}，跳过。")
            continue

        # 收集所有 md 文件
        all_md_files = sorted([
            f for f in os.listdir(md_dir) if f.lower().endswith(".md")
        ])

        if not all_md_files:
            print(f"[{dataset}] md 目录为空，跳过。")
            continue

        # 断点恢复: 过滤已处理文件
        done_files = load_done_files(dataset)
        pending_files = [f for f in all_md_files if os.path.splitext(f)[0] not in done_files]

        # 测试模式: 随机抽取 3 个
        if MODE == "test":
            pending_files = random.sample(pending_files, min(3, len(pending_files)))

        print(f"\n[{dataset}] 总计: {len(all_md_files)} | 已完成: {len(done_files)} | 本次处理: {len(pending_files)} | 模式: {MODE}")

        if not pending_files:
            print(f"[{dataset}] 无待处理文件。")
            continue

        # 测试模式输出文件（含思维链）
        test_output_file = os.path.join(dataset_dir, f"{dataset}_test_output.json")

        # 逐文件处理（stage-1/stage-2 各自通过独立信号量控制并发）
        async def _worker(md_filename):
            md_path = os.path.join(md_dir, md_filename)
            return await process_single_file(md_path, dataset, dataset_dir, sem1, sem2)

        tasks = [_worker(f) for f in pending_files]

        total_valid = 0
        total_invalid = 0
        total_errors = 0
        total_no_data = 0

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"[{dataset}] 提取进度"):
            result = await coro
            filename = result["filename"]

            if result["error"]:
                print(f"  [WARN] {filename}: {result['error']}")
                total_errors += 1
                # 即使出错也标记为已处理，避免反复卡住
                mark_done(dataset, filename)
                continue

            final_entries = result["final_entries"] or []

            # 无数据提取情况：记录文件名，跳过后续处理
            if not final_entries:
                total_no_data += 1
                log_no_data(dataset, filename)
                # 测试模式仍保存思维链以便调试
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

            # 为每个条目添加 filename 字段（添加在 doi 下方）
            for entry in final_entries:
                if isinstance(entry, dict):
                    # 重新构建 dict 以确保 filename 在 doi 之后
                    new_entry = {}
                    for k, v in entry.items():
                        new_entry[k] = v
                        if k == "doi":
                            new_entry["filename"] = filename
                    if "filename" not in new_entry:
                        new_entry["filename"] = filename
                    entry.clear()
                    entry.update(new_entry)

            # 格式检查与分流
            v, iv = check_and_route(final_entries, dataset, dataset_dir, filename)
            total_valid += v
            total_invalid += iv

            # 测试模式: 保存思维链
            if MODE == "test":
                test_record = {
                    "filename": filename,
                    "stage1_entries": result["stage1_entries"],
                    "stage1_reasoning": result["stage1_reasoning"],
                    "stage2_entries": result["stage2_entries"],
                    "stage2_reasoning": result["stage2_reasoning"],
                    "final_entries": final_entries,
                }
                # 追加写入
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

            # 标记完成
            mark_done(dataset, filename)

        print(f"[{dataset}] 完成。合格: {total_valid} | 不合格: {total_invalid} | 无数据: {total_no_data} | 错误: {total_errors}")


if __name__ == "__main__":
    # 必须在 asyncio.run() 之前设置，否则对当前循环无效
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
