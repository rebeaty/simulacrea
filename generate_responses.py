"""
ITCT Validation Study - LLM Response Generation
================================================
Generates creativity task responses using various LLM backends.
Supports: CPO, Gemini Flash, Base Llama, Centaur-70B

Generation approach:
- Each participant completes ALL tasks in ONE batch API call (session cohesion)
- 1 response per item (32 total items per participant)
- Responses parsed from batch output using task markers
"""

import asyncio
import aiohttp
import json
import csv
import random
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import re

# Import task specifications
from TASK_SPECIFICATION import (
    ALL_TASKS, AUT_ITEMS, DESIGN_ITEMS,
    SCTT_HYPOTHESIS_ITEMS, SCTT_RESEARCH_QUESTION_ITEMS,
    STORY_THEMES, METAPHOR_ITEMS,
    DAT_PROMPT_TEMPLATE, STORY_PROMPT_TEMPLATE, METAPHOR_PROMPT_TEMPLATE,
    CREATIVE_ACHIEVEMENTS_PROMPT_TEMPLATE,
    AUT_INSTRUCTION, DESIGN_INSTRUCTION, SCTT_INSTRUCTION
)

# ══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

STANDARD_PARAMS = {
    "max_tokens": 1200,     # 16 tasks @ ~75 tokens each (matches working script)
    "temperature": 0.7,
    "top_p": 0.95,
}

MODEL_CONFIGS = {
    "cpo": {
        "api_url": "https://mkdbkqb2enqnr59l.us-east-1.aws.endpoints.huggingface.cloud",
        "api_key_env": "HF_TOKEN",
    },
    "gemini": {
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
        "api_key_env": "GOOGLE_API_KEY",
    },
    "llama_base": {
        "api_url": "https://tjoaf6n5vwtjthkp.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
        "api_key_env": "HF_TOKEN",
    },
    "centaur": {
        "api_url": "https://jyvw17pus60ll41l.us-east-2.aws.endpoints.huggingface.cloud/generate",
        "api_key_env": "HF_TOKEN",
    }
}

SAMPLE_SIZE = 245
RANDOMIZE_ORDER = True

RATE_LIMITS = {
    "centaur": 4,
    "gemini": 100,
    "cpo": 8,
    "llama_base": 10,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#                           BUILD TASK LIST
# ══════════════════════════════════════════════════════════════════════════════

def build_task_list() -> List[Dict]:
    """Build list of 16 individual tasks (matches working script structure)
    4 AUT + 4 Design + 4 Story + 4 SCTT = 16 tasks
    """
    tasks = []

    # AUT - 4 items (first 4)
    for item in AUT_ITEMS[:4]:
        tasks.append({
            "task": "AUT",
            "item": item,
            "prompt": (
                "Think of a clever, unusual, interesting, uncommon, humorous, innovative, or "
                "different use for the following object. Your response should be a few words to "
                f"a sentence MAX.\n\nSCENARIO: {item}\n\nGenerate one response and be creative!"
            )
        })

    # Design - 4 items (first 4)
    for item in DESIGN_ITEMS[:4]:
        item_short = item.replace("Develop as many design ideas as you can to ", "")
        tasks.append({
            "task": "Design",
            "item": item_short,
            "full_item": item,
            "prompt": (
                "Think of a solution to a real-world design problem. Generate an idea that is original "
                "and effective. Your response should be a few words to a sentence MAX.\n\n"
                f"SCENARIO: {item_short}\n\nGenerate one response and be creative!"
            )
        })

    # Stories - 4 items (first 4)
    story_items = list(STORY_THEMES.items())[:4]
    for theme_id, theme_data in story_items:
        words_str = "-".join(theme_data["words"])
        tasks.append({
            "task": "Story",
            "item": theme_id,
            "prompt": (
                "Write a very short story using the provided words. Your story should be 4-5 short sentences MAX."
                f"\n\nSCENARIO: {words_str}\n\nGenerate one response and be creative!"
            )
        })

    # SCTT - 2 Hypothesis + 2 Research Questions = 4 items
    for item in SCTT_HYPOTHESIS_ITEMS[:2]:
        tasks.append({
            "task": "SCTT-Hypothesis",
            "item": item[:50],
            "full_item": item,
            "prompt": (
                "Think of an original, scientifically plausible hypothesis to explain the following scenario. "
                f"Your response should be a single sentence MAX.\n\nSCENARIO: {item}\n\n"
                "Generate one response and be creative!"
            )
        })

    for item in SCTT_RESEARCH_QUESTION_ITEMS[:2]:
        tasks.append({
            "task": "SCTT-ResearchQuestion",
            "item": item[:50],
            "full_item": item,
            "prompt": (
                "Think of an original, scientifically plausible question about the following scenario. "
                f"Your response should be a single sentence MAX.\n\nSCENARIO: {item}\n\n"
                "Generate one response and be creative!"
            )
        })

    return tasks

# ══════════════════════════════════════════════════════════════════════════════
#                           BATCH PROMPT BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_batch_prompt(tasks: List[Dict], model: str = "default") -> str:
    """Build a single batch prompt containing all tasks with markers"""
    if model == "centaur":
        # Centaur needs simpler format without ###TASK### markers
        header = "Complete these creative tasks. Give one brief response per task.\n\n"
        blocks = [header]
        for idx, t in enumerate(tasks, 1):
            blocks.append(f"Task {idx}: {t['prompt']}")
        blocks.append("\nResponses:")
        return "\n\n".join(blocks)
    else:
        # Standard format for CPO, Llama, Gemini
        header = (
            "You will complete multiple creative tasks.\n"
            "For EACH task, respond with the task marker (e.g., '###TASK_1###') on its own line, "
            "followed by your answer, then '###END###' on its own line.\n"
            "Output ONLY your responses - no extra commentary.\n\n"
        )
        blocks = [header]
        for idx, t in enumerate(tasks, 1):
            blocks.append(f"###TASK_{idx}###")
            blocks.append(t["prompt"])
        return "\n\n".join(blocks)

def parse_batch_response(text: str, num_tasks: int, model: str = "default") -> Dict[int, str]:
    """Parse batch response to extract individual task responses"""
    parsed = {}

    # Centaur uses simpler format: "Task N: response text"
    if model == "centaur":
        # Simple pattern: Task N: followed by response (until next Task or end)
        # Format: "Task 1: response text\nTask 2: response text\n..."
        task_pattern = r'Task\s*(\d+)\s*:\s*(.+?)(?=Task\s*\d+\s*:|$)'
        matches = re.findall(task_pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            try:
                task_num = int(match[0])
                content = match[1].strip()
                # Take first line as response (handles multi-line content)
                first_line = content.split('\n')[0].strip()
                if first_line and task_num not in parsed:
                    parsed[task_num] = first_line
            except (ValueError, IndexError):
                continue
        return parsed

    # Try standard pattern first
    task_pattern = r'###TASK_(\d+)###'
    matches = list(re.finditer(task_pattern, text))

    if not matches:
        # Try alternate patterns
        task_pattern = r'###TASK[_\s]*(\d+)[_\s]*###'
        matches = list(re.finditer(task_pattern, text))

    for i, match in enumerate(matches):
        try:
            task_num = int(match.group(1))
            start_pos = match.end()

            # Find end position
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_match = re.search(r'###END###', text[start_pos:])
                if end_match:
                    end_pos = start_pos + end_match.start()
                else:
                    end_pos = len(text)

            content = text[start_pos:end_pos].strip()
            content = re.sub(r'###END###\s*$', '', content).strip()

            if task_num not in parsed and content:
                parsed[task_num] = content

        except (ValueError, AttributeError):
            continue

    return parsed

# ══════════════════════════════════════════════════════════════════════════════
#                         API GENERATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

async def generate_cpo(prompt: str, max_retries: int = 3) -> str:
    """Generate using CPO endpoint - fresh session per call (matches working script)"""
    config = MODEL_CONFIGS["cpo"]
    headers = {
        "Authorization": f"Bearer {os.environ.get(config['api_key_env'], '')}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": [prompt],
        "parameters": {
            "max_new_tokens": STANDARD_PARAMS["max_tokens"],
            "temperature": STANDARD_PARAMS["temperature"],
            "top_p": STANDARD_PARAMS["top_p"],
            "do_sample": True,
            "repetition_penalty": 1.1
        }
    }

    for attempt in range(max_retries):
        try:
            # Fresh session per call - matches working script pattern
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config["api_url"],
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0].get("response", "").strip()
                    elif response.status == 503:
                        logger.warning(f"CPO 503 - endpoint starting up, waiting...")
                        await asyncio.sleep(30)
                    else:
                        error_text = await response.text()
                        logger.warning(f"CPO status {response.status}: {error_text[:200]}")
                        await asyncio.sleep(5)
        except Exception as e:
            logger.warning(f"CPO attempt {attempt+1} failed: {e}")
            await asyncio.sleep(5)
    return ""

async def generate_gemini(prompt: str, max_retries: int = 3) -> str:
    """Generate using Gemini - fresh session per call"""
    config = MODEL_CONFIGS["gemini"]
    api_key = os.environ.get(config["api_key_env"], "")
    url = f"{config['api_url']}?key={api_key}"

    # Gemini 3 docs recommend temperature 1.0 for best results
    # "changing the temperature (setting it below 1.0) may lead to unexpected behavior"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 8000,  # Much higher for Gemini (verbose)
            "temperature": 1.0,  # Per Gemini 3 docs - required for proper behavior
            "topP": STANDARD_PARAMS["top_p"]
        }
    }

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "candidates" in result and len(result["candidates"]) > 0:
                            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
                    elif response.status == 429:
                        logger.warning(f"Gemini 429 rate limit, waiting...")
                        await asyncio.sleep(30 * (attempt + 1))
                    else:
                        error_text = await response.text()
                        logger.warning(f"Gemini status {response.status}: {error_text[:200]}")
                        await asyncio.sleep(5)
        except Exception as e:
            logger.warning(f"Gemini attempt {attempt+1} failed: {e}")
            await asyncio.sleep(5)
    return ""

async def generate_llama_base(prompt: str, max_retries: int = 3) -> str:
    """Generate using Llama Base via chat completions - fresh session per call"""
    config = MODEL_CONFIGS["llama_base"]
    headers = {
        "Authorization": f"Bearer {os.environ.get(config['api_key_env'], '')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "tgi",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": STANDARD_PARAMS["temperature"],
        "top_p": STANDARD_PARAMS["top_p"],
        "max_tokens": STANDARD_PARAMS["max_tokens"]
    }

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config["api_url"],
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    elif response.status == 503:
                        logger.warning(f"Llama 503 - endpoint starting up, waiting...")
                        await asyncio.sleep(30)
                    else:
                        error_text = await response.text()
                        logger.warning(f"Llama status {response.status}: {error_text[:200]}")
                        await asyncio.sleep(5)
        except Exception as e:
            logger.warning(f"Llama attempt {attempt+1} failed: {e}")
            await asyncio.sleep(5)
    return ""

async def generate_centaur(prompt: str, max_retries: int = 3) -> str:
    """Generate using Centaur - fresh session per call"""
    config = MODEL_CONFIGS["centaur"]
    headers = {
        "Authorization": f"Bearer {os.environ.get(config['api_key_env'], '')}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": STANDARD_PARAMS["max_tokens"],
            "temperature": STANDARD_PARAMS["temperature"],
            "top_p": STANDARD_PARAMS["top_p"],
            "do_sample": True,
            "return_full_text": False,
            "repetition_penalty": 1.1
        }
    }

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config["api_url"],
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        text = ""
                        if isinstance(result, list) and len(result) > 0:
                            text = result[0].get("generated_text", "").strip()
                        elif isinstance(result, dict):
                            text = result.get("generated_text", "").strip()
                        if not text:
                            logger.warning(f"Centaur empty response. Raw: {str(result)[:300]}")
                        return text
                    elif response.status == 503:
                        logger.warning(f"Centaur 503 - endpoint starting up, waiting...")
                        await asyncio.sleep(30)
                    else:
                        error_text = await response.text()
                        logger.warning(f"Centaur status {response.status}: {error_text[:200]}")
                        await asyncio.sleep(5)
        except Exception as e:
            logger.warning(f"Centaur attempt {attempt+1} failed: {e}")
            await asyncio.sleep(5)
    return ""

async def generate_response(prompt: str, model: str) -> str:
    """Route to appropriate generation function (no session - each creates its own)"""
    if model == "cpo":
        return await generate_cpo(prompt)
    elif model == "gemini":
        return await generate_gemini(prompt)
    elif model == "llama_base":
        return await generate_llama_base(prompt)
    elif model == "centaur":
        return await generate_centaur(prompt)
    else:
        logger.error(f"Unknown model: {model}")
        return ""

# ══════════════════════════════════════════════════════════════════════════════
#                           MAIN GENERATION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

async def run_generation(model: str, output_file: str, sample_size: int = None, start_from: int = 1, append: bool = False):
    """Run generation for a single model - SEQUENTIAL processing (matches working script)"""
    if sample_size is None:
        sample_size = SAMPLE_SIZE

    # Build task list (25 items)
    tasks = build_task_list()
    participants = [f"{model}_{i:03d}" for i in range(start_from, sample_size + 1)]

    if not participants:
        logger.info(f"No participants to process (start_from={start_from}, sample_size={sample_size})")
        return None

    logger.info(f"ITCT Validation Generation (Sequential Batch Mode)")
    logger.info(f"  Model: {model}")
    logger.info(f"  Participants: {start_from} to {sample_size} ({len(participants)} total)")
    logger.info(f"  Tasks per participant: {len(tasks)}")
    logger.info(f"  Total responses: {len(participants) * len(tasks)}")
    logger.info(f"  API calls: {len(participants)} (1 batch per participant, sequential)")
    logger.info(f"  Mode: {'APPEND' if append else 'OVERWRITE'}")

    # Check API key
    config = MODEL_CONFIGS[model]
    api_key = os.environ.get(config["api_key_env"], "")
    if not api_key:
        logger.error(f"Missing API key: {config['api_key_env']}")
        return None

    # Setup CSV (append or overwrite)
    file_mode = "a" if append else "w"
    csv_file = open(output_file, file_mode, newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["entity_id", "prompt", "response", "task", "source", "word_count"]
    )
    if not append:
        csv_writer.writeheader()

    results = []
    start_time = time.time()
    stamp = int(time.time())

    with open(f"raw_api_responses_{model}_{stamp}.log", "w", encoding="utf-8") as rawlog:
        # Process participants SEQUENTIALLY (like working script)
        for i, pid in enumerate(participants, 1):
            logger.info(f"\n{'='*60}\n[{model}] Processing {pid} ({i}/{len(participants)})\n{'='*60}")

            # Shuffle task order for this participant
            order = tasks.copy()
            if RANDOMIZE_ORDER:
                random.shuffle(order)

            # Build batch prompt (Centaur uses different format)
            batch_prompt = build_batch_prompt(order, model)

            # Make single API call (each function creates its own fresh session)
            try:
                raw_response = await generate_response(batch_prompt, model)
            except Exception as e:
                logger.error(f"Error for {pid}: {e}")
                raw_response = ""

            # Log raw response
            rawlog.write(f"\n--- RAW REPLY {pid} ---\n{raw_response}\n")
            rawlog.flush()

            # Parse responses (Centaur uses different parser)
            parsed = parse_batch_response(raw_response, len(order), model)

            # Create and save results for each task
            for idx, t in enumerate(order, 1):
                response = parsed.get(idx, "")
                if not response:
                    logger.warning(f"{pid} missing response for task {idx} ({t['task']})")

                word_count = len(response.split()) if response else 0
                result = {
                    "entity_id": pid,
                    "prompt": t.get("full_item", t["item"]),
                    "response": response,
                    "task": t["task"],
                    "source": model,
                    "word_count": word_count
                }
                results.append(result)
                csv_writer.writerow(result)

            csv_file.flush()  # Write immediately

            # Show preview of responses
            non_empty = sum(1 for idx in range(1, len(order)+1) if parsed.get(idx))
            logger.info(f"{pid}: {non_empty}/{len(order)} tasks captured")

            # Sleep between participants (like working script)
            await asyncio.sleep(1)

    csv_file.close()
    elapsed = time.time() - start_time

    logger.info(f"\n[{model}] Generation complete!")
    logger.info(f"  Total responses: {len(results)}")
    non_empty_count = sum(1 for r in results if r["response"])
    logger.info(f"  Non-empty: {non_empty_count}/{len(results)} ({100*non_empty_count/len(results):.1f}%)")
    logger.info(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  CSV: {output_file}")

    # Task breakdown
    from collections import Counter
    task_counts = Counter(r["task"] for r in results)
    logger.info(f"\n[{model}] Task breakdown:")
    for task, count in sorted(task_counts.items()):
        non_empty_task = sum(1 for r in results if r["task"] == task and r["response"])
        logger.info(f"  {task}: {non_empty_task}/{count}")

    return results

async def run_all_models_parallel(models: List[str], output_dir: str, sample_size: int = None):
    """Run all models in parallel"""
    if sample_size is None:
        sample_size = SAMPLE_SIZE

    logger.info("=" * 60)
    logger.info("PARALLEL MODEL GENERATION (Batch Mode)")
    logger.info("=" * 60)
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Tasks per participant: 32")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Check API keys
    missing = []
    for model in models:
        config = MODEL_CONFIGS[model]
        if not os.environ.get(config["api_key_env"], ""):
            missing.append(f"{model}: {config['api_key_env']}")
    if missing:
        logger.error("Missing API keys:")
        for m in missing:
            logger.error(f"  {m}")
        return

    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for model in models:
        output_file = os.path.join(output_dir, f"{model}_responses.csv")
        tasks.append(run_generation(model, output_file, sample_size))

    start_time = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("ALL MODELS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    for model, result in zip(models, results):
        if result:
            logger.info(f"  {model}: {len(result)} responses")
        else:
            logger.info(f"  {model}: FAILED")
    logger.info("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
#                                  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate ITCT responses (batch mode)")
    parser.add_argument("--model", choices=["cpo", "gemini", "llama_base", "centaur"])
    parser.add_argument("--models", nargs="+", choices=["cpo", "gemini", "llama_base", "centaur"])
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="llm_responses")
    parser.add_argument("--sample-size", type=int, default=245)
    parser.add_argument("--start-from", type=int, default=1, help="Start from participant N (for continuing/topping up)")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")

    args = parser.parse_args()

    if args.all_models:
        models = ["cpo", "gemini", "llama_base", "centaur"]
        asyncio.run(run_all_models_parallel(models, args.output_dir, args.sample_size))
    elif args.models:
        asyncio.run(run_all_models_parallel(args.models, args.output_dir, args.sample_size))
    elif args.model:
        output_file = args.output or f"{args.model}_responses.csv"
        asyncio.run(run_generation(args.model, output_file, args.sample_size, args.start_from, args.append))
    else:
        parser.error("Must specify --model, --models, or --all-models")

if __name__ == "__main__":
    main()
