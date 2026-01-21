"""
Creativity Score Distribution Analysis
Compares creativity scores between human and LLM responses across tasks.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# ================== CONFIG ==================
BASE_PATH = Path(__file__).parent
TASK_CSV_PATH = BASE_PATH / "task_csvs_for_cap"
OUTPUT_DIR = BASE_PATH / "analysis_results"

# ================== LOAD DATA ==================
def load_scored_data():
    """Load scored data for all tasks"""
    aut = pd.read_csv(TASK_CSV_PATH / "aut_cap_scored.csv")
    sctt = pd.read_csv(TASK_CSV_PATH / "sctt_cap_scored.csv")
    design = pd.read_csv(TASK_CSV_PATH / "design_cap_scored.csv")
    story_dsi = pd.read_csv(TASK_CSV_PATH / "story_cap_dsi_scored.csv")
    story_maoss = pd.read_csv(TASK_CSV_PATH / "story_cap_maoss_scored.csv")

    print(f"Loaded data:")
    print(f"  AUT: {len(aut)} rows, sources: {aut['source'].value_counts().to_dict()}")
    print(f"  SCTT: {len(sctt)} rows, sources: {sctt['source'].value_counts().to_dict()}")
    print(f"  Design: {len(design)} rows, sources: {design['source'].value_counts().to_dict()}")
    print(f"  Story DSI: {len(story_dsi)} rows, sources: {story_dsi['source'].value_counts().to_dict()}")
    print(f"  Story MAoSS: {len(story_maoss)} rows, sources: {story_maoss['source'].value_counts().to_dict()}")

    return aut, sctt, design, story_dsi, story_maoss

# ================== ANALYSIS ==================
def analyze_distribution(df, task_name, score_col='prediction'):
    """Analyze score distribution by source with statistical tests"""
    print(f"\n{'='*60}")
    print(f"{task_name}")
    print(f"{'='*60}")

    sources = ['human', 'cpo', 'llama_base', 'gemini']
    human_data = df[df['source'] == 'human'][score_col].dropna()

    # Descriptive stats
    print(f"\n{'Source':<12} | {'n':>6} | {'mean':>6} | {'std':>6}")
    print("-" * 40)

    results = []
    for source in sources:
        data = df[df['source'] == source][score_col].dropna()
        if len(data) > 0:
            print(f"{source:<12} | {len(data):>6} | {data.mean():>6.3f} | {data.std():>6.3f}")

            if source != 'human':
                ks_stat, ks_p = stats.ks_2samp(human_data, data)
                pooled_std = np.sqrt((human_data.std()**2 + data.std()**2) / 2)
                cohens_d = (human_data.mean() - data.mean()) / pooled_std if pooled_std > 0 else 0
                results.append({
                    'task': task_name,
                    'source': source,
                    'n': len(data),
                    'mean': data.mean(),
                    'std': data.std(),
                    'ks_stat': ks_stat,
                    'ks_p': ks_p,
                    'cohens_d': cohens_d
                })

    # KS tests
    print(f"\nKS tests vs human:")
    for r in results:
        sig = "*" if r['ks_p'] < 0.05 else ""
        print(f"  {r['source']:<12} KS={r['ks_stat']:.3f}, d={r['cohens_d']:.2f}{sig}")

    return pd.DataFrame(results)

def plot_distributions(df, task_name, output_dir, score_col='prediction'):
    """Create distribution plot"""
    fig, ax = plt.subplots(figsize=(10, 5))

    sources = ['human', 'cpo', 'llama_base', 'gemini']
    colors = {'human': '#2ecc71', 'cpo': '#e74c3c', 'llama_base': '#3498db', 'gemini': '#9b59b6'}

    for source in sources:
        data = df[df['source'] == source][score_col].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.4,
                   label=f'{source} (n={len(data)}, μ={data.mean():.3f})',
                   color=colors.get(source, 'gray'), density=True)

    ax.set_xlabel('Creativity Score')
    ax.set_ylabel('Density')
    ax.set_title(f'{task_name} - Score Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'distribution_{task_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()

# ================== FILTERING ==================
def filter_extreme_values(df, score_col='prediction', lower_percentile=1, upper_percentile=99):
    """Filter extreme values based on percentiles"""
    lower = df[score_col].quantile(lower_percentile / 100)
    upper = df[score_col].quantile(upper_percentile / 100)
    original_len = len(df)
    filtered = df[(df[score_col] >= lower) & (df[score_col] <= upper)]
    removed = original_len - len(filtered)
    if removed > 0:
        print(f"  Filtered {removed} extreme values (keeping {lower_percentile}-{upper_percentile} percentile)")
    return filtered

# ================== MAIN ==================
def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("CREATIVITY SCORE ANALYSIS")
    print("="*60)

    aut, sctt, design, story_dsi, story_maoss = load_scored_data()

    all_results = []

    # AUT (CAP)
    results = analyze_distribution(aut, "AUT (CAP Originality)")
    plot_distributions(aut, "AUT", OUTPUT_DIR)
    all_results.append(results)

    # SCTT (CAP)
    results = analyze_distribution(sctt, "SCTT (CAP Originality)")
    plot_distributions(sctt, "SCTT", OUTPUT_DIR)
    all_results.append(results)

    # Design Originality (CAP)
    results = analyze_distribution(design, "Design Originality (CAP)")
    plot_distributions(design, "Design_Originality", OUTPUT_DIR)
    all_results.append(results)

    # Design Effectiveness (CAP)
    if 'effectiveness_prediction' in design.columns:
        results = analyze_distribution(design, "Design Effectiveness (CAP)",
                                       score_col='effectiveness_prediction')
        plot_distributions(design, "Design_Effectiveness", OUTPUT_DIR,
                          score_col='effectiveness_prediction')
        all_results.append(results)

    # Story (DSI) - filter extreme values
    print("\nFiltering Story DSI extreme values...")
    story_dsi_filtered = filter_extreme_values(story_dsi)
    results = analyze_distribution(story_dsi_filtered, "Story (DSI)")
    plot_distributions(story_dsi_filtered, "Story_DSI", OUTPUT_DIR)
    all_results.append(results)

    # Story (MAoSS/AI) - filter extreme values
    print("\nFiltering Story MAoSS extreme values...")
    story_maoss_filtered = filter_extreme_values(story_maoss)
    results = analyze_distribution(story_maoss_filtered, "Story (MAoSS)")
    plot_distributions(story_maoss_filtered, "Story_MAoSS", OUTPUT_DIR)
    all_results.append(results)

    # Save combined results
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / 'creativity_scores_summary.csv', index=False)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
