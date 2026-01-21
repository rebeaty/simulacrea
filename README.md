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
| Gemini | Gemini 2.0 Flash | 245 |

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
├── preprint.tex                 # LaTeX preprint
├── generate_responses.py        # LLM response generation
├── create_matched_samples.py    # Match human samples to LLM structure
├── create_figures.py            # Generate publication figures
└── analyze_creativity_scores.py # Distribution analysis
```

## Usage

```bash
# Generate LLM responses (requires API keys)
python generate_responses.py --model crpo --output llm_responses_batch/crpo_responses.csv

# Create matched human samples
python create_matched_samples.py

# Analyze creativity score distributions
python analyze_creativity_scores.py

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
