import os
import glob
import json
import asyncio
from openai import AsyncOpenAI
import platform
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

load_dotenv()

# ================= Configuration =================
CONCURRENCY = 64
MAX_RETRIES = 3
TARGET_DATASETS = ["E"] # "Tg", "Tm", "n", "eps", "E", "UTS"
MODE = "prod" # "test" or "prod"

MODEL_NAME = "qwen3.5-plus-2026-02-15"

# Ignore progress file, force reprocess all entries (use with caution)
IGNORE_PROGRESS = False

# Working root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =================================================

client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

PROMPT_TASK1 = """
# Role
You are an elite Polymer Physicist and Materials Scientist. Your task is to analyze structured data of a polymer sample and transform it into fluent natural language and deeply reasoned scientific analysis.

# Task
I will provide you with a JSON object containing the data of a single polymer sample (including its chemical composition, processing history, and hierarchical structure. Note: The actual property values have been intentionally omitted). 
Your objective is to read this data and generate a strictly formatted JSON output containing two fields:
1. A fluent, coherent natural language description of the polymer.
2. A logical reasoning text analyzing how the polymer's composition, structure, AND processing history influence its {TARGET_PROPERTY}.

# Rules & Constraints (CRITICAL)

1. **"descriptive_text"**:
   - Synthesize the `chemical_composition` (including monomers AND any additives such as fillers, plasticizers, metals, etc.), `processing_history`, and `hierarchical_structure` into a **single, fluent, and cohesive paragraph**.
   - Do NOT use bullet points. Make it read like the "Experimental" or "Material Characterization" section of a high-impact scientific journal.
   - Smoothly handle any `null` values or empty arrays by omitting them; do not explicitly mention missing data.

2. **"reasoning_analysis"**:
   - **NO VALUE PREDICTION**: You are STRICTLY FORBIDDEN from predicting, estimating, or guessing the exact numerical value or range of the {TARGET_PROPERTY}. Focus EXCLUSIVELY on qualitative directional influence.
   - Based on polymer physics principles, analyze how the specific monomers, additives (if any), structural features, AND the specific `processing_history` (e.g., thermal treatment, solvent effects, cooling rates, characterization methods) logically affect the {TARGET_PROPERTY}.
   - You must structure your text using exactly these three aspects:
     a) Increasing Factors: Specific factors driving the {TARGET_PROPERTY} UP.
     b) Decreasing Factors: Specific factors driving the {TARGET_PROPERTY} DOWN.
     c) Dominant Factor: Which factor qualitatively dominates the final {TARGET_PROPERTY} behavior for this specific sample.

# Output Format
You MUST output **ONLY** a valid JSON object. Do NOT include any Markdown formatting blocks (e.g., ```json) or conversational text. 

# Example

User Input (Assume {TARGET_PROPERTY} is "Glass Transition Temperature (Tg)"):
{
  "chemical_composition": {
    "repeat_unit_psmiles": "[*]CC(C1=CC=CC=C1)[*]",
    "monomers": [{"name": "Styrene", "smiles": "C=CC1=CC=CC=C1", "ratio_value": null, "ratio_unit": null}],
    "additives": [{"name": "Shimalite NAW-101 silicate powder", "type": "filler", "amount_value": null, "amount_unit": null}],
    "is_homopolymer": true
  },
  "processing_history": "Sample prepared as powder. Dissolved in benzene at room temperature, mixed with silicate powder, and then dried in vacuum. Tg measured by DSC at 5 °C/min under N2 atmosphere.",
  "hierarchical_structure": "Amorphous, monodisperse, Mp = 7,600 g/mol"
}

Expected JSON Output:
{
  "descriptive_text": "The sample is an amorphous, monodisperse polystyrene homopolymer with a molecular weight (Mp) of 7,600 g/mol. It was prepared by dissolving the polymer in benzene at room temperature, followed by mixing with Shimalite NAW-101 silicate powder and subsequent vacuum drying to yield a powder composite. The glass transition temperature was characterized by DSC at 5 °C/min under N2 atmosphere.",
  "reasoning_analysis": "Increasing Factors: The presence of bulky phenyl side groups in the styrene repeating units restricts polymer chain mobility through steric hindrance, intrinsically driving the Tg higher. Additionally, the processing history involving mixing with rigid silicate powder may restrict segmental motion at the polymer-filler interface, further contributing to an increase in Tg. Decreasing Factors: The relatively low molecular weight (7,600 g/mol) increases the free volume at chain ends (Fox-Flory effect), which enhances chain mobility and lowers the Tg. Furthermore, if the vacuum drying process in the processing history did not completely remove the benzene solvent, residual benzene could act as a plasticizer, significantly depressing the Tg. Dominant Factor: The intrinsic rigidity provided by the bulky phenyl groups establishes a high baseline Tg for the polymer backbone; however, the low molecular weight and potential residual solvent from the ambient processing history are likely the dominant factors causing a qualitative reduction in Tg relative to pristine, high-molecular-weight polystyrene."
}

# Input Data
Here is the JSON data of the polymer sample to process (Target Property: {TARGET_PROPERTY}):
{INPUT_JSON_DATA}
"""

PROMPT_TASK2 = """
# Role
You are an elite Polymer Physicist and Materials Scientist. Your task is to analyze structured data of a polymer sample and transform it into fluent natural language and deeply reasoned scientific analysis.

# Task
I will provide you with a JSON object containing the data of a single polymer sample (specifically its chemical composition and processing history. Note: Hierarchical structure and actual property values have been intentionally omitted). 
Your objective is to read this data and generate a strictly formatted JSON output containing two fields:
1. A fluent, coherent natural language description of the polymer.
2. A logical reasoning text analyzing how the polymer's composition AND processing history influence its {TARGET_PROPERTY}.

# Rules & Constraints (CRITICAL)

1. **"descriptive_text"**:
   - Synthesize the `chemical_composition` (including monomers AND any additives such as fillers, plasticizers, metals, etc.) and `processing_history` into a **single, fluent, and cohesive paragraph**.
   - Do NOT use bullet points. Make it read like the "Experimental" or "Material Preparation" section of a high-impact scientific journal.
   - Smoothly handle any `null` values or empty arrays by omitting them; do not explicitly mention missing data.

2. **"reasoning_analysis"**:
   - **NO VALUE PREDICTION**: You are STRICTLY FORBIDDEN from predicting, estimating, or guessing the exact numerical value or range of the {TARGET_PROPERTY}. Focus EXCLUSIVELY on qualitative directional influence.
   - Based on polymer physics principles, analyze how the specific monomers, additives (if any), polymer architecture (derived from composition), AND the specific `processing_history` (e.g., thermal treatment, solvent effects, blending, drying methods, characterization methods) logically affect the {TARGET_PROPERTY}.
   - You must structure your text using exactly these three aspects:
     a) Increasing Factors: Specific factors driving the {TARGET_PROPERTY} UP.
     b) Decreasing Factors: Specific factors driving the {TARGET_PROPERTY} DOWN.
     c) Dominant Factor: Which factor qualitatively dominates the final {TARGET_PROPERTY} behavior for this specific sample.

# Output Format
You MUST output **ONLY** a valid JSON object. Do NOT include any Markdown formatting blocks (e.g., ```json) or conversational text. 

# Example

User Input (Assume {TARGET_PROPERTY} is "Glass Transition Temperature (Tg)"):
{
  "chemical_composition": {
    "repeat_unit_psmiles": "[*]CC(C1=CC=CC=C1)[*]",
    "monomers": [{"name": "Styrene", "smiles": "C=CC1=CC=CC=C1", "ratio_value": null, "ratio_unit": null}],
    "additives": [{"name": "Shimalite NAW-101 silicate powder", "type": "filler", "amount_value": null, "amount_unit": null}],
    "is_homopolymer": true
  },
  "processing_history": "Sample prepared as powder. Dissolved in benzene at room temperature, mixed with silicate powder, and then dried in vacuum. Tg measured by DSC at 5 °C/min under N2 atmosphere."
}

Expected JSON Output:
{
  "descriptive_text": "The sample is a polystyrene homopolymer. It was prepared by dissolving the polymer in benzene at room temperature, followed by mixing with Shimalite NAW-101 silicate powder and subsequent vacuum drying to yield a powder composite. The glass transition temperature was measured by DSC at 5 °C/min under N2 atmosphere.",
  "reasoning_analysis": "Increasing Factors: The inherent chemical composition, specifically the presence of bulky phenyl side groups on the polystyrene backbone, restricts polymer chain mobility through steric hindrance, intrinsically driving the Tg higher. Additionally, the processing history involving mixing with rigid silicate powder likely restricts segmental motion at the polymer-filler interface, further contributing to an increase in Tg. Decreasing Factors: If the vacuum drying process in the processing history did not completely remove the benzene solvent, residual benzene molecules could act as a plasticizer, significantly increasing free volume and depressing the Tg. Dominant Factor: The intrinsic structural rigidity provided by the bulky phenyl groups establishes a high baseline Tg for the polymer backbone; however, the potential plasticization effect from residual solvent due to the ambient processing history is the dominant dynamic factor that could cause a qualitative reduction in Tg relative to a pristine, fully dried sample."
}

# Input Data
Here is the JSON data of the polymer sample to process (Target Property: {TARGET_PROPERTY}):
{INPUT_JSON_DATA}
"""


# ===========================================================================
# Progress Management (Checkpoint Resume)
# ===========================================================================
def _progress_file(dataset: str) -> str:
    return os.path.join(BASE_DIR, "final", f"{dataset}_processor_progress.txt")


def load_done_indices(dataset: str) -> set:
    """Load set of completed entry indices. Returns empty set if IGNORE_PROGRESS=True."""
    if IGNORE_PROGRESS:
        return set()
    pf = _progress_file(dataset)
    if not os.path.exists(pf):
        return set()
    with open(pf, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip()}


def mark_done_index(dataset: str, idx: int):
    """Mark an entry index as processed."""
    pf = _progress_file(dataset)
    with open(pf, "a", encoding="utf-8") as f:
        f.write(str(idx) + "\n")


def load_existing_results(output_file: str) -> list:
    """Load existing output results."""
    if not os.path.exists(output_file):
        return []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, IOError):
        pass
    return []


async def call_llm_with_retry(prompt: str, sem: asyncio.Semaphore) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                response = await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=MODEL_NAME,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": True}
                )
                
                content = response.choices[0].message.content or ""
                msg_dict = response.choices[0].message.model_dump()
                reasoning = msg_dict.get("reasoning_content", "")
                
                # remove optional markdown formatting if present
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                    
                parsed = json.loads(content)
                return {
                    "result": parsed,
                    "reasoning": reasoning
                }
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Error after {MAX_RETRIES} attempts: {e}")
                return {"result": None, "reasoning": ""}
            await asyncio.sleep(1)
            
    return {"result": None, "reasoning": ""}


async def process_single_item(item: dict, dataset_name: str, sem: asyncio.Semaphore) -> dict:
    # Task 1 payload
    input_data_1 = {
        "chemical_composition": item.get("chemical_composition"),
        "processing_history": item.get("processing_history"),
        "hierarchical_structure": item.get("hierarchical_structure")
    }
    
    # Task 2 payload
    input_data_2 = {
        "chemical_composition": item.get("chemical_composition"),
        "processing_history": item.get("processing_history")
    }
    
    prompt1 = PROMPT_TASK1.replace("{TARGET_PROPERTY}", dataset_name).replace("{INPUT_JSON_DATA}", json.dumps(input_data_1, ensure_ascii=False))
    prompt2 = PROMPT_TASK2.replace("{TARGET_PROPERTY}", dataset_name).replace("{INPUT_JSON_DATA}", json.dumps(input_data_2, ensure_ascii=False))
    
    task1, task2 = await asyncio.gather(
        call_llm_with_retry(prompt1, sem),
        call_llm_with_retry(prompt2, sem)
    )
    
    # attach outputs to the original item
    item["analysis_with_hierarchical_structure"] = task1["result"]
    item["analysis_without_hierarchical_structure"] = task2["result"]
    
    if MODE == "test":
        item["reasoning_chain_with_hierarchical_structure"] = task1["reasoning"]
        item["reasoning_chain_without_hierarchical_structure"] = task2["reasoning"]
        
    return item

async def main():
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    os.makedirs("final", exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    
    for dataset in TARGET_DATASETS:
        folder_path = dataset
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} doesn't exist, skipping.")
            continue
        
        # 读取带 title 的文件（由 03_add_title.py 生成）
        input_file = os.path.join(folder_path, f"{dataset}_extracted_titled.json")
        if not os.path.exists(input_file):
            print(f"No titled JSON file found: {input_file}, skipping.")
            continue
        final_file = os.path.join("final", f"{dataset}.json")
        
        print(f"Processing {dataset} dataset from {input_file} (Mode: {MODE})")
        
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if MODE == "test":
            data = data[:3]
        
        # Checkpoint resume: load completed indices and existing results
        done_indices = load_done_indices(dataset)
        results = load_existing_results(final_file)
        
        # Filter pending entries (preserve original indices)
        pending = [(i, item) for i, item in enumerate(data) if i not in done_indices]
        
        print(f"[{dataset}] Total: {len(data)} | Done: {len(done_indices)} | Processing this run: {len(pending)}")
        
        if not pending:
            print(f"[{dataset}] No entries to process.")
            continue
        
        # Create async tasks for each pending entry, carrying index info
        async def _worker(idx, item):
            result = await process_single_item(item, dataset, sem)
            result["_idx"] = idx
            return result
        
        tasks = [_worker(i, item) for i, item in pending]
        
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {dataset}"):
            res = await coro
            idx = res.pop("_idx")
            results.append(res)
            
            # Write complete results to output file (ensures file is always valid JSON)
            with open(final_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Mark this index as done
            mark_done_index(dataset, idx)
                
        print(f"Dataset {dataset} saved to {final_file} with {len(results)} records.")

if __name__ == "__main__":
    asyncio.run(main())
