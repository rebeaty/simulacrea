# simulacrea

**Simulating Creativity: LLMs as Synthetic Participants in Creativity Research**

This repository contains code and analysis for comparing LLM-generated creativity responses against human responses on established psychometric tasks.

## Overview

We evaluate whether LLMs can serve as valid "simulacra" for creativity research by comparing response distributions from three model configurations against ~240 human participants across four creativity tasks.

## Tasks

- **AUT** (Alternate Uses Task): Generate creative uses for common objects
- **SCTT** (Scientific Creative Thinking Task): Generate hypotheses and research questions
- **Design**: Propose solutions to real-world design problems
- **Story**: Write short creative stories incorporating given words

## Models

| Model | Description | n |
|-------|-------------|---|
| Human | Prolific participants | ~240 |
| CRPO | Creative Preference Optimization (Llama 3.1 fine-tuned on MuCE dataset) | 245 |
| Llama Base | Llama 3.1 8B | 245 |
| Gemini | Gemini 3 Flash | 245 |
| Centaur-70B | Llama 3.1 70B fine-tuned on Psych-101 (marcelbinz/Llama-3.1-Centaur-70B) | 245 |
| Minitaur-8B | Llama 3.1 8B fine-tuned on Psych-101 (marcelbinz/Llama-3.1-Minitaur-8B) | 245 |

Centaur/Minitaur are *cognitive foundation models* trained to mimic human
trial-by-trial behavior; they require Psych-101-style transcript prompting and a
dedicated generation script — see [docs/RUNNING_CENTAUR.md](docs/RUNNING_CENTAUR.md).

## Scoring

- **CAP** (Creative Assessment Platform): AI-based originality, effectiveness, and story creativity scoring
- **DSI** (Divergent Semantic Integration): Average pairwise semantic distance between words

## Key Findings

- **CRPO produces the most human-like creativity distributions** across 4 of 6 metrics
- CRPO matches human effectiveness ratings almost perfectly (d = 0.03)
- CRPO responses show highest semantic similarity to human response centroids
- Preference-optimized training yields more valid simulacra than base models

See [analysis_results/REPORT_MATCHED.md](analysis_results/REPORT_MATCHED.md) for detailed results.

## Project Structure

```
├── human_data/                  # Human response data
├── llm_responses_batch/         # Generated LLM responses
├── task_csvs_for_cap/           # Scored data files
├── analysis_results/            # Analysis outputs and report
├── figures/                     # Publication figures
├── docs/                        # Centaur deployment guide, open-dataset survey
├── preprint.tex                 # LaTeX preprint
├── TASK_SPECIFICATION.py        # Task items and prompt templates
├── generate_responses.py        # LLM response generation (chat models)
├── generate_centaur_responses.py# Centaur/Minitaur generation (Psych-101 format)
├── create_matched_samples.py    # Match human samples to LLM structure
├── create_figures.py            # Generate publication figures
├── analyze_creativity_scores.py # Distribution analysis
└── analyze_covariance.py        # Cross-task correlation structure vs humans
```

## Usage

```bash
# Generate LLM responses (requires API keys)
python generate_responses.py --model crpo --output llm_responses_batch/crpo_responses.csv

# Generate Centaur/Minitaur responses (see docs/RUNNING_CENTAUR.md for backends)
python generate_centaur_responses.py --model centaur_70b --backend hf-router
python generate_centaur_responses.py --model centaur_8b --backend vllm

# Create matched human samples
python create_matched_samples.py

# Analyze creativity score distributions
python analyze_creativity_scores.py

# Cross-task correlation structure (human vs each model)
python analyze_covariance.py

# Generate publication figures
python create_figures.py
```

## Requirements

```bash
pip install -r requirements.txt
```

## Citation

If you use this code or data, please cite:

```bibtex
@article{simulacrea2025,
  title={Simulating Creativity: LLMs as Synthetic Participants in Creativity Research},
  author={Anonymous},
  year={2025}
}
```

## License

MIT
