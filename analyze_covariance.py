"""
Cross-task covariance analysis: does each model reproduce the HUMAN pattern of
within-participant correlations between creativity tasks?

For each source (human, cpo, llama_base, gemini, centaur_70b, centaur_8b, ...):
  1. compute each participant's mean score per task from the scored CAP CSVs
  2. correlate task scores across participants (task x task correlation matrix)
  3. compare each model's matrix to the human matrix:
       - profile similarity: Pearson r between the off-diagonal elements
       - mean absolute deviation of corresponding correlations
  4. item-level difficulty profile: per-item mean score, human vs model

Outputs:
  analysis_results/task_correlations_<source>.csv  (per-source matrices)
  analysis_results/covariance_summary.csv          (model-vs-human similarity)
  analysis_results/item_profile_summary.csv        (item mean profile match)
  analysis_results/task_correlation_heatmaps.png
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_PATH = Path(__file__).parent
TASK_CSV_PATH = BASE_PATH / "task_csvs_for_cap"
OUTPUT_DIR = BASE_PATH / "analysis_results"

MIN_N = 30  # minimum participants for a correlation to be reported


def load_task_scores():
    """Return long dataframe: id, source, task_metric, item, score."""
    frames = []

    aut = pd.read_csv(TASK_CSV_PATH / "aut_cap_scored.csv")
    frames.append(aut.assign(task_metric="AUT_originality", score=aut["prediction"]))

    sctt = pd.read_csv(TASK_CSV_PATH / "sctt_cap_scored.csv")
    frames.append(sctt.assign(task_metric="SCTT_originality", score=sctt["prediction"]))

    design = pd.read_csv(TASK_CSV_PATH / "design_cap_scored.csv")
    frames.append(design.assign(task_metric="Design_originality", score=design["prediction"]))
    if "effectiveness_prediction" in design.columns:
        frames.append(design.assign(task_metric="Design_effectiveness",
                                    score=design["effectiveness_prediction"]))

    story_dsi = pd.read_csv(TASK_CSV_PATH / "story_cap_dsi_scored.csv")
    frames.append(story_dsi.assign(task_metric="Story_DSI", score=story_dsi["prediction"]))

    story_maoss = pd.read_csv(TASK_CSV_PATH / "story_cap_maoss_scored.csv")
    frames.append(story_maoss.assign(task_metric="Story_MAoSS", score=story_maoss["prediction"]))

    cols = ["id", "source", "task_metric", "item", "score"]
    return pd.concat([f[cols] for f in frames], ignore_index=True).dropna(subset=["score"])


def participant_task_means(long_df):
    """participant x task_metric mean score table, per source."""
    return (long_df
            .groupby(["source", "id", "task_metric"])["score"].mean()
            .unstack("task_metric"))


def correlation_matrix(wide_df):
    return wide_df.corr(min_periods=MIN_N)


def offdiag_vector(mat, order):
    pairs = list(itertools.combinations(order, 2))
    return np.array([mat.loc[a, b] for a, b in pairs]), pairs


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    long_df = load_task_scores()
    by_participant = participant_task_means(long_df)

    sources = list(by_participant.index.get_level_values("source").unique())
    # human first, then everything else in a stable order
    sources = ["human"] + sorted(s for s in sources if s != "human")

    matrices = {}
    for source in sources:
        wide = by_participant.loc[source]
        mat = correlation_matrix(wide)
        matrices[source] = mat
        mat.to_csv(OUTPUT_DIR / f"task_correlations_{source}.csv")
        print(f"\n=== {source} (n={len(wide)}) task intercorrelations ===")
        print(mat.round(2).to_string())

    task_order = [t for t in matrices["human"].columns]

    # Model vs human similarity of correlation structure
    human_vec, pairs = offdiag_vector(matrices["human"], task_order)
    rows = []
    for source in sources:
        if source == "human":
            continue
        vec, _ = offdiag_vector(matrices[source], task_order)
        valid = ~(np.isnan(human_vec) | np.isnan(vec))
        if valid.sum() < 3:
            continue
        profile_r = np.corrcoef(human_vec[valid], vec[valid])[0, 1]
        mad = np.mean(np.abs(human_vec[valid] - vec[valid]))
        rows.append({"source": source,
                     "n_pairs": int(valid.sum()),
                     "profile_r_vs_human": profile_r,
                     "mean_abs_dev": mad})
        print(f"\n{source}: correlation-profile r vs human = {profile_r:.3f}, "
              f"mean |Δr| = {mad:.3f}")

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "covariance_summary.csv", index=False)

    # Item-level difficulty/mean profile match
    item_means = (long_df.groupby(["source", "task_metric", "item"])["score"]
                  .mean().rename("mean_score").reset_index())
    human_items = item_means[item_means.source == "human"]
    item_rows = []
    for source in sources:
        if source == "human":
            continue
        merged = human_items.merge(
            item_means[item_means.source == source],
            on=["task_metric", "item"], suffixes=("_human", "_model"))
        if len(merged) >= 3:
            r = np.corrcoef(merged["mean_score_human"], merged["mean_score_model"])[0, 1]
            item_rows.append({"source": source, "n_items": len(merged),
                              "item_profile_r_vs_human": r})
            print(f"{source}: item mean-profile r vs human = {r:.3f} ({len(merged)} items)")
    pd.DataFrame(item_rows).to_csv(OUTPUT_DIR / "item_profile_summary.csv", index=False)

    # Heatmaps
    n = len(sources)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
    for ax, source in zip(axes[0], sources):
        mat = matrices[source].loc[task_order, task_order]
        im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(task_order)))
        ax.set_xticklabels(task_order, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(task_order)))
        ax.set_yticklabels(task_order, fontsize=7)
        ax.set_title(source)
        for i in range(len(task_order)):
            for j in range(len(task_order)):
                v = mat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    plt.savefig(OUTPUT_DIR / "task_correlation_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
