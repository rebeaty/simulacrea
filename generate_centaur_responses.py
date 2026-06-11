"""
Centaur / Minitaur Response Generation (Psych-101 format)
=========================================================
Generates creativity-task responses from Centaur-70B and Minitaur-8B
(marcelbinz/Llama-3.1-Centaur-70B, marcelbinz/Llama-3.1-Minitaur-8B).

Why a separate script: Centaur is NOT an instruction-tuned chat model. It is
fine-tuned on Psych-101 — natural-language transcripts of psychology
experiments in which the participant's actions are wrapped in << >> tokens.
Prompting it with generic instruction batches (the approach in
generate_responses.py) makes it echo the prompt instead of responding, which
is what happened in the original llm_responses_batch/centaur_responses.csv.

Correct usage, per the Centaur paper (arXiv:2410.20268):
  - describe the experiment in second person, trial by trial
  - end the prompt with `You respond <<` (prefill)
  - sample until the closing `>>`

Each synthetic participant is one *session*: tasks are presented sequentially
in randomized order and the model's earlier responses stay in context, exactly
like a Psych-101 multi-trial transcript. This preserves within-participant
coherence, which matters when testing participant-level correlations.

Backends (choose with --backend):
  vllm       OpenAI-compatible completions API (default http://localhost:8000/v1).
             Serve with: vllm serve marcelbinz/Llama-3.1-Minitaur-8B
             See docs/RUNNING_CENTAUR.md for 70B options.
  hf-router  Hugging Face Inference Providers router (needs HF_TOKEN).
             Centaur-70B is live on the featherless-ai provider, so the 70B
             can be run with no GPU of your own.
  tgi        A dedicated HF Inference Endpoint (raw TGI /generate API).

Examples:
  python generate_centaur_responses.py --model centaur_70b --backend hf-router
  python generate_centaur_responses.py --model centaur_8b --backend vllm
  python generate_centaur_responses.py --model centaur_70b --dry-run
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import requests

from TASK_SPECIFICATION import (
    AUT_ITEMS, DESIGN_ITEMS, STORY_THEMES,
    SCTT_HYPOTHESIS_ITEMS, SCTT_RESEARCH_QUESTION_ITEMS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "centaur_70b": "marcelbinz/Llama-3.1-Centaur-70B",
    "centaur_8b": "marcelbinz/Llama-3.1-Minitaur-8B",
}

# featherless-ai currently hosts the 70B; the 8B has no serverless provider
HF_ROUTER_URL = "https://router.huggingface.co/v1/completions"
HF_ROUTER_PROVIDER = "featherless-ai"

SAMPLE_SIZE = 245
RANDOMIZE_ORDER = True

GEN_PARAMS = {
    "temperature": 1.0,   # match human variability; Centaur simulations sample at T=1
    "top_p": 0.95,
}
MAX_TOKENS = {"Story": 200, "default": 80}
STOP_SEQUENCES = [">>"]

# ══════════════════════════════════════════════════════════════════════════════
#                       PSYCH-101 STYLE PROMPT CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

PREAMBLE = (
    "You are participating in a study of creative thinking. You will complete a "
    "series of short creative tasks. For each task, you read the instructions and "
    "then type your response.\n\n"
)


def build_task_list():
    """16 trials: 4 AUT + 4 Design + 4 Story + 2 SCTT-H + 2 SCTT-RQ.

    Same items as the original generate_responses.py runs ([:4] slices), so
    Centaur output is directly comparable to the existing cpo / llama_base /
    gemini / human matched samples. The `prompt` field written to the CSV uses
    the same item identifiers as the other response files.
    """
    tasks = []

    for item in AUT_ITEMS[:4]:
        tasks.append({
            "task": "AUT",
            "item": item,
            "trial": (
                "You are asked to think of a clever, unusual, interesting, uncommon, "
                "humorous, innovative, or different use for an everyday object. Your "
                f"response should be a few words to a sentence at most. The object is "
                f"{item}. You respond <<"
            ),
        })

    for item in DESIGN_ITEMS[:4]:
        item_short = item.replace("Develop as many design ideas as you can to ", "").rstrip(".")
        tasks.append({
            "task": "Design",
            "item": item,
            "trial": (
                "You are asked to propose an original and effective solution to a "
                "real-world design problem. Your response should be a few words to a "
                f"sentence at most. The problem is to {item_short}. You respond <<"
            ),
        })

    for theme_id, theme in list(STORY_THEMES.items())[:4]:
        w = theme["words"]
        tasks.append({
            "task": "Story",
            "item": theme_id,
            "trial": (
                "You are asked to write a very short creative story, four to five "
                f"short sentences at most, that uses three given words. The words are "
                f"{w[0]}, {w[1]}, and {w[2]}. You respond <<"
            ),
        })

    for item in SCTT_HYPOTHESIS_ITEMS[:2]:
        tasks.append({
            "task": "SCTT-Hypothesis",
            "item": item,
            "trial": (
                "You are asked to come up with an original, scientifically plausible "
                "hypothesis about a scenario. Your response should be a single "
                f"sentence at most. The scenario is: {item} You respond <<"
            ),
        })

    for item in SCTT_RESEARCH_QUESTION_ITEMS[:2]:
        tasks.append({
            "task": "SCTT-ResearchQuestion",
            "item": item,
            "trial": (
                "You are asked to come up with an original scientific question about "
                "a scenario. Your response should be a single sentence at most. The "
                f"scenario is: {item} You respond <<"
            ),
        })

    return tasks


def clean_response(text: str) -> str:
    """Trim generation to the participant's response inside << >>."""
    text = text.split(">>")[0]          # in case the stop sequence was not applied
    text = text.replace("<<", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ══════════════════════════════════════════════════════════════════════════════
#                                BACKENDS
# ══════════════════════════════════════════════════════════════════════════════


def complete_vllm(prompt: str, model_id: str, max_tokens: int, base_url: str) -> str:
    r = requests.post(
        f"{base_url.rstrip('/')}/completions",
        json={
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": GEN_PARAMS["temperature"],
            "top_p": GEN_PARAMS["top_p"],
            "stop": STOP_SEQUENCES,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def complete_hf_router(prompt: str, model_id: str, max_tokens: int, base_url: str) -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is required for --backend hf-router")
    r = requests.post(
        HF_ROUTER_URL,
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": f"{model_id}:{HF_ROUTER_PROVIDER}",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": GEN_PARAMS["temperature"],
            "top_p": GEN_PARAMS["top_p"],
            "stop": STOP_SEQUENCES,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def complete_tgi(prompt: str, model_id: str, max_tokens: int, base_url: str) -> str:
    token = os.environ.get("HF_TOKEN", "")
    r = requests.post(
        f"{base_url.rstrip('/')}/generate",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": GEN_PARAMS["temperature"],
                "top_p": GEN_PARAMS["top_p"],
                "do_sample": True,
                "stop": STOP_SEQUENCES,
                "return_full_text": False,
            },
        },
        timeout=300,
    )
    r.raise_for_status()
    out = r.json()
    if isinstance(out, list):
        out = out[0]
    return out.get("generated_text", "")


BACKENDS = {"vllm": complete_vllm, "hf-router": complete_hf_router, "tgi": complete_tgi}
DEFAULT_BASE_URLS = {"vllm": "http://localhost:8000/v1", "hf-router": "", "tgi": ""}


def complete_with_retry(backend, prompt, model_id, max_tokens, base_url,
                        max_retries=5) -> str:
    delay = 2
    for attempt in range(max_retries):
        try:
            return BACKENDS[backend](prompt, model_id, max_tokens, base_url)
        except Exception as e:
            logger.warning(f"attempt {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return ""

# ══════════════════════════════════════════════════════════════════════════════
#                                GENERATION
# ══════════════════════════════════════════════════════════════════════════════


def run_participant(pid: str, tasks, backend, model_id, base_url, rawlog=None):
    """Run one synthetic participant as a single accumulating session transcript."""
    order = tasks.copy()
    if RANDOMIZE_ORDER:
        random.shuffle(order)

    transcript = PREAMBLE
    rows = []
    for t in order:
        prompt = transcript + t["trial"]
        max_tokens = MAX_TOKENS.get(t["task"], MAX_TOKENS["default"])
        raw = complete_with_retry(backend, prompt, model_id, max_tokens, base_url)
        response = clean_response(raw)
        if rawlog:
            rawlog.write(f"\n--- {pid} | {t['task']} | {t['item'][:60]} ---\n{raw}\n")
            rawlog.flush()
        # close the trial in the transcript so the next trial sees the history
        transcript += t["trial"] + f"{response}>>. "
        rows.append({
            "entity_id": pid,
            "prompt": t["item"],
            "response": response,
            "task": t["task"],
            "source": None,  # filled by caller
            "word_count": len(response.split()) if response else 0,
        })
    return rows


def run_generation(model_key, backend, base_url, output_file, sample_size,
                   start_from, append, dry_run, seed):
    model_id = MODELS[model_key]
    tasks = build_task_list()
    random.seed(seed)

    if dry_run:
        order = tasks.copy()
        random.shuffle(order)
        transcript = PREAMBLE
        print("=" * 70)
        print(f"DRY RUN — example session transcript for {model_id}")
        print("=" * 70)
        for t in order[:3]:
            print(transcript + t["trial"] + "[GENERATED RESPONSE]>>.\n")
            transcript += t["trial"] + "[GENERATED RESPONSE]>>. "
            print("-" * 70)
        print(f"... ({len(order)} trials total per participant)")
        return

    participants = [f"{model_key}_{i:03d}" for i in range(start_from, sample_size + 1)]
    logger.info(f"Centaur generation | model={model_id} backend={backend}")
    logger.info(f"  participants {start_from}..{sample_size} | {len(tasks)} trials each | "
                f"{len(participants) * len(tasks)} responses total")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    stamp = int(time.time())

    with open(out_path, mode, newline="", encoding="utf-8") as f, \
         open(f"raw_api_responses_{model_key}_{stamp}.log", "w", encoding="utf-8") as rawlog:
        writer = csv.DictWriter(
            f, fieldnames=["entity_id", "prompt", "response", "task", "source", "word_count"])
        if not append:
            writer.writeheader()

        start = time.time()
        for i, pid in enumerate(participants, 1):
            rows = run_participant(pid, tasks, backend, model_id, base_url, rawlog)
            n_ok = sum(1 for r in rows if r["response"])
            for r in rows:
                r["source"] = model_key
                writer.writerow(r)
            f.flush()
            logger.info(f"[{i}/{len(participants)}] {pid}: {n_ok}/{len(rows)} responses "
                        f"({time.time() - start:.0f}s elapsed)")

    logger.info(f"Done -> {output_file}")


def main():
    p = argparse.ArgumentParser(description="Generate Centaur/Minitaur creativity responses")
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--backend", choices=list(BACKENDS), default="vllm")
    p.add_argument("--base-url", default=None,
                   help="API base URL (default: http://localhost:8000/v1 for vllm; "
                        "endpoint URL required for tgi)")
    p.add_argument("--output", default=None,
                   help="output CSV (default: llm_responses_batch/<model>_responses.csv)")
    p.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    p.add_argument("--start-from", type=int, default=1)
    p.add_argument("--append", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="print example prompts, no API calls")
    args = p.parse_args()

    base_url = args.base_url or DEFAULT_BASE_URLS[args.backend]
    if args.backend == "tgi" and not base_url:
        p.error("--base-url is required for --backend tgi")
    output = args.output or f"llm_responses_batch/{args.model}_responses.csv"

    run_generation(args.model, args.backend, base_url, output,
                   args.sample_size, args.start_from, args.append,
                   args.dry_run, args.seed)


if __name__ == "__main__":
    main()
