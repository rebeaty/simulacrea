# Running Centaur-70B and Minitaur-8B

Centaur (`marcelbinz/Llama-3.1-Centaur-70B`) and its 8B sibling Minitaur
(`marcelbinz/Llama-3.1-Minitaur-8B`) are Llama-3.1 models fine-tuned (QLoRA) on
[Psych-101](https://huggingface.co/datasets/marcelbinz/Psych-101), 10.7M choices
from 60k participants in 160 psychology experiments.

**They are completion models, not chat models.** Prompts must read like a
Psych-101 experiment transcript (second person, trial by trial) and end with
`You respond <<` (or `You press <<`); generation must stop at `>>`.
`generate_centaur_responses.py` handles all of this — the only decision is
where the model runs.

Total workload per model: 245 participants x 16 trials = **3,920 completions**,
each ~200-2,000 prompt tokens (the session transcript grows) and <=200 output
tokens. This is a small job: a few GPU-hours on the 70B, well under one
GPU-hour on the 8B.

---

## Option A — 70B with no GPU at all (recommended first step)

Centaur-70B is hosted by the **featherless-ai** serverless provider on
Hugging Face. Any HF account with inference credits can use it through the
HF router — no endpoint deployment needed:

```bash
export HF_TOKEN=hf_...   # token with "Make calls to Inference Providers" permission
python generate_centaur_responses.py --model centaur_70b --backend hf-router
```

Notes:
- Uses `https://router.huggingface.co/v1/completions` with model
  `marcelbinz/Llama-3.1-Centaur-70B:featherless-ai`.
- Featherless is rate-limited per-account concurrency; the script runs
  sequentially per participant, so expect several hours wall-clock. Use
  `--start-from N --append` to resume after interruptions.
- Minitaur-8B is **not** on any serverless provider — for the 8B use Option B
  or C.

## Option B — vLLM on your own GPU(s)

### Minitaur-8B (any 24 GB GPU)

```bash
pip install vllm
vllm serve marcelbinz/Llama-3.1-Minitaur-8B --max-model-len 8192
python generate_centaur_responses.py --model centaur_8b --backend vllm
```

bf16 weights are ~16 GB; fits a single RTX 3090/4090/A5000/L4(24GB).

### Centaur-70B

Pick based on available hardware (bf16 weights are ~140 GB):

| Hardware | Command |
|---|---|
| 1x 80 GB (A100/H100) | `vllm serve marcelbinz/Llama-3.1-Centaur-70B --quantization bitsandbytes --load-format bitsandbytes --max-model-len 8192` (in-flight 4-bit, ~40 GB; slower per-token but fine for 4k calls) |
| 2x 80 GB | `vllm serve marcelbinz/Llama-3.1-Centaur-70B --tensor-parallel-size 2 --max-model-len 8192` (full bf16) |
| 4x 48 GB (A6000/L40S) | `vllm serve marcelbinz/Llama-3.1-Centaur-70B --tensor-parallel-size 4 --max-model-len 8192` |

Then:

```bash
python generate_centaur_responses.py --model centaur_70b --backend vllm
```

### Adapter-on-quantized-base alternative (1x 48 GB)

The official model card's low-memory recipe applies the LoRA adapter to a
4-bit base with unsloth. If you only have a single 48 GB card:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="marcelbinz/Llama-3.1-Centaur-70B-adapter",
    max_seq_length=32768,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

(vLLM cannot serve a LoRA adapter on top of a bitsandbytes-quantized base, so
this path means transformers/unsloth generation rather than the vLLM server.
If you go this way, it is usually simpler to use Option A for the 70B and
reserve local GPUs for the 8B.)

## Option C — dedicated HF Inference Endpoint (what the earlier run used)

Deploy `marcelbinz/Llama-3.1-Centaur-70B` (TGI, e.g. 2x A100) or
`marcelbinz/Llama-3.1-Minitaur-8B` (1x A10G/L4) as an Inference Endpoint, then:

```bash
export HF_TOKEN=hf_...
python generate_centaur_responses.py --model centaur_70b --backend tgi \
    --base-url https://<your-endpoint>.endpoints.huggingface.cloud
```

> The earlier Centaur run (archived in
> `llm_responses_batch/archive/centaur_responses_invalid_prompt_format.csv`)
> used a working endpoint but the wrong prompt format — Centaur echoed the
> scenarios back instead of responding. The endpoint was fine; the prompts
> were not. `generate_centaur_responses.py` fixes this.

---

## Sanity check before a full run

Generate 3 participants and eyeball the responses:

```bash
python generate_centaur_responses.py --model centaur_70b --backend hf-router --sample-size 3
head -20 llm_responses_batch/centaur_70b_responses.csv
```

Good output = short, human-plausible answers ("hold a door open", "use it as a
plant holder"), no prompt echoes, no instruction-following boilerplate.

## After generation: scoring and analysis

1. `python create_matched_samples.py` — regenerates `task_csvs_for_cap/*_for_cap_matched.csv`
   including the new `centaur_70b` / `centaur_8b` sources automatically.
2. Score the new responses with CAP (originality/effectiveness) and DSI, producing
   updated `task_csvs_for_cap/*_cap_scored.csv` with the same columns as before.
3. `python analyze_creativity_scores.py` — distribution matching (KS, Cohen's d),
   new sources picked up automatically.
4. `python analyze_covariance.py` — cross-task correlation structure and
   item-profile match vs humans (the key new test).
5. `python create_figures.py` — publication figures.
