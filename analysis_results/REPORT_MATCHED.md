# Human vs LLM Creativity Comparison

## Overview

This analysis compares human responses to LLM responses (CRPO, Llama Base, Gemini) on creativity tasks using **matched samples** (one response per item per participant).

**Models:**
- **CRPO**: Creative Preference Optimization (Llama 3.1 fine-tuned on human creativity data)
- **Llama Base**: Llama 3.1 8B with persona prompting
- **Gemini**: Gemini 3 Flash with persona prompting

## Sample Sizes

| Source | AUT | Design | Story | SCTT | Total |
|--------|-----|--------|-------|------|-------|
| Human | 1,191 | 1,158 | 1,149 | 1,185 | 4,683 |
| CRPO | 1,032 | 1,032 | 1,032 | 1,032 | 4,128 |
| Llama Base | 1,004 | 1,004 | 1,004 | 1,004 | 4,016 |
| Gemini | 980 | 980 | 980 | 980 | 3,920 |

---

## Creativity Score Distributions

### AUT Originality (CAP)

| Source | Mean | Std | Cohen's d vs Human |
|--------|------|-----|-------------------|
| Human | 0.521 | 0.085 | — |
| **CRPO** | 0.564 | 0.091 | **-0.49** |
| Llama Base | 0.592 | 0.078 | -0.87 |
| Gemini | 0.650 | 0.076 | -1.59 |

### SCTT Originality (CAP)

| Source | Mean | Std | Cohen's d vs Human |
|--------|------|-----|-------------------|
| Human | 0.528 | 0.125 | — |
| **CRPO** | 0.617 | 0.111 | **-0.75** |
| Llama Base | 0.686 | 0.079 | -1.51 |
| Gemini | 0.722 | 0.130 | -1.53 |

### Design Originality (CAP)

| Source | Mean | Std | Cohen's d vs Human |
|--------|------|-----|-------------------|
| Human | 0.471 | 0.146 | — |
| **CRPO** | 0.603 | 0.151 | **-0.89** |
| Llama Base | 0.717 | 0.105 | -1.94 |
| Gemini | 0.758 | 0.131 | -2.07 |

### Design Effectiveness (CAP)

| Source | Mean | Std | Cohen's d vs Human |
|--------|------|-----|-------------------|
| Human | 0.535 | 0.129 | — |
| **CRPO** | **0.532** | 0.167 | **0.03** (ns) |
| Llama Base | 0.693 | 0.121 | -1.26 |
| Gemini | 0.622 | 0.139 | -0.65 |

### Story Creativity (DSI)

| Source | Mean | Std | Cohen's d vs Human |
|--------|------|-----|-------------------|
| **Human** | **0.796** | 0.028 | — |
| CRPO | 0.750 | 0.195 | 0.33 |
| **Llama Base** | 0.781 | 0.106 | **0.20** |
| Gemini | 0.756 | 0.164 | 0.34 |

### Story Creativity (MAoSS/AI-rated)

| Source | Mean | Std | Cohen's d vs Human |
|--------|------|-----|-------------------|
| Human | 0.423 | 0.116 | — |
| CRPO | 0.513 | 0.131 | -0.72 |
| Llama Base | 0.490 | 0.091 | -0.64 |
| **Gemini** | **0.470** | 0.091 | **-0.45** |

---

## Key Findings

### CAP-Scored Tasks (AUT, SCTT, Design)
- All LLMs score **higher** than humans on AI-rated originality
- **CRPO is closest** to human levels (smallest |Cohen's d|)
- Gemini scores highest but is furthest from human distribution

### Story Tasks
- **DSI** (semantic distance): Humans score higher; **Llama Base** closest (d=0.20)
- **MAoSS** (AI-rated): LLMs score higher; **Gemini** closest (d=-0.45)
- Different metrics yield opposing conclusions

### Design Effectiveness
- **CRPO matches human distribution almost perfectly** (d = 0.03)
- Only metric where a model is statistically near-equivalent to humans

---

## Summary: Which Model is Most Human-Like?

| Metric | Most Human-Like | Least Human-Like |
|--------|-----------------|------------------|
| AUT Originality | **CRPO** | Gemini |
| SCTT Originality | **CRPO** | Gemini |
| Design Originality | **CRPO** | Gemini |
| Design Effectiveness | **CRPO** (d=0.03) | Llama Base |
| Story (DSI) | **Llama Base** | Gemini |
| Story (MAoSS) | **Gemini** | CRPO |

### Overall: **CRPO**
- Closest to human on 4/6 metrics
- Best at matching human effectiveness distribution

---

## Interpretation

The divergence between scoring metrics suggests they capture different aspects of creativity:

- **CAP/MAoSS** (AI-rated): rewards unusual/original ideas → LLMs excel
- **DSI** (semantic distance): measures pairwise word distance → Humans excel

This raises questions about what AI-based creativity metrics actually measure and whether they inadvertently favor LLM-typical outputs.

---

*Report generated from matched samples (one response per item per participant, random seed=42)*
