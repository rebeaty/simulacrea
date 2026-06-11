"""
Generate publication-quality figures for the preprint.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# ================== CONFIG ==================
BASE_PATH = Path(__file__).parent
TASK_CSV_PATH = BASE_PATH / "task_csvs_for_cap"
ANALYSIS_PATH = BASE_PATH / "analysis_results"
OUTPUT_DIR = BASE_PATH / "figures"

# Style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

COLORS = {
    'human': '#2ecc71',      # Green
    'cpo': '#e74c3c',        # Red (CRPO)
    'llama_base': '#3498db', # Blue
    'gemini': '#9b59b6',     # Purple
    'centaur_70b': '#f39c12',# Orange
    'centaur_8b': '#16a085'  # Teal
}

LABELS = {
    'human': 'Human',
    'cpo': 'CRPO',
    'llama_base': 'Llama Base',
    'gemini': 'Gemini',
    'centaur_70b': 'Centaur-70B',
    'centaur_8b': 'Minitaur-8B'
}

def model_sources(df):
    """Non-human sources present in the data, in canonical order."""
    present = set(df['source'].dropna().unique())
    return [s for s in LABELS if s != 'human' and s in present]

# ================== LOAD DATA ==================
def load_data():
    """Load all scored data"""
    aut = pd.read_csv(TASK_CSV_PATH / "aut_cap_scored.csv")
    sctt = pd.read_csv(TASK_CSV_PATH / "sctt_cap_scored.csv")
    design = pd.read_csv(TASK_CSV_PATH / "design_cap_scored.csv")
    story_dsi = pd.read_csv(TASK_CSV_PATH / "story_cap_dsi_scored.csv")
    story_maoss = pd.read_csv(TASK_CSV_PATH / "story_cap_maoss_scored.csv")

    return {
        'AUT': aut,
        'SCTT': sctt,
        'Design': design,
        'Story (DSI)': story_dsi,
        'Story (MAoSS)': story_maoss
    }

# ================== FIGURE 1: Distribution Overlaps ==================
def create_distribution_figure(data):
    """Create a multi-panel figure showing distribution overlaps for key tasks"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    tasks = [
        ('AUT', 'prediction', 'AUT Originality'),
        ('Design', 'prediction', 'Design Originality'),
        ('Design', 'effectiveness_prediction', 'Design Effectiveness'),
        ('Story (MAoSS)', 'prediction', 'Story Creativity (MAoSS)')
    ]

    for idx, (task_key, score_col, title) in enumerate(tasks):
        ax = axes[idx]
        df = data[task_key]
        sources = ['human'] + model_sources(df)

        for source in sources:
            source_data = df[df['source'] == source][score_col].dropna()
            if len(source_data) > 0:
                # KDE plot
                sns.kdeplot(
                    data=source_data,
                    ax=ax,
                    color=COLORS[source],
                    label=LABELS[source],
                    linewidth=2,
                    fill=True,
                    alpha=0.2
                )

        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig1_distributions.pdf/png")

# ================== FIGURE 2: Effect Size Summary ==================
def create_effect_size_figure(data):
    """Create a bar chart showing Cohen's d for each task/model combination with bootstrap CIs"""

    # Calculate effect sizes with bootstrap confidence intervals
    def bootstrap_cohens_d(human_data, source_data, n_bootstrap=1000):
        """Calculate |Cohen's d| with bootstrap 95% CI"""
        ds = []
        n_human = len(human_data)
        n_source = len(source_data)
        human_arr = human_data.values
        source_arr = source_data.values

        for _ in range(n_bootstrap):
            h_sample = np.random.choice(human_arr, n_human, replace=True)
            s_sample = np.random.choice(source_arr, n_source, replace=True)
            pooled_std = np.sqrt((h_sample.std()**2 + s_sample.std()**2) / 2)
            if pooled_std > 0:
                ds.append(abs((h_sample.mean() - s_sample.mean()) / pooled_std))

        d_mean = np.mean(ds)
        d_ci = np.percentile(ds, [2.5, 97.5])
        return d_mean, d_ci[0], d_ci[1]

    results = []
    tasks_configs = [
        ('AUT', 'prediction', 'AUT'),
        ('SCTT', 'prediction', 'SCTT'),
        ('Design', 'prediction', 'Design Orig.'),
        ('Design', 'effectiveness_prediction', 'Design Eff.'),
        ('Story (MAoSS)', 'prediction', 'Story')
    ]

    for task_key, score_col, label in tasks_configs:
        df = data[task_key]
        human_data = df[df['source'] == 'human'][score_col].dropna()

        for source in model_sources(df):
            source_data = df[df['source'] == source][score_col].dropna()
            if len(source_data) > 0:
                d_mean, d_lo, d_hi = bootstrap_cohens_d(human_data, source_data)
                results.append({
                    'Task': label,
                    'Model': LABELS[source],
                    'Cohen_d': d_mean,
                    'CI_lo': d_lo,
                    'CI_hi': d_hi
                })

    results_df = pd.DataFrame(results)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 5))

    tasks = ['AUT', 'SCTT', 'Design Orig.', 'Design Eff.', 'Story']
    plotted_sources = [s for s in LABELS if s != 'human'
                       and LABELS[s] in set(results_df['Model'])]
    x = np.arange(len(tasks))
    width = 0.8 / max(len(plotted_sources), 1)

    for i, source in enumerate(plotted_sources):
        model = LABELS[source]
        model_data = results_df[results_df['Model'] == model]
        values = [model_data[model_data['Task'] == t]['Cohen_d'].values[0] for t in tasks]
        ci_lo = [model_data[model_data['Task'] == t]['CI_lo'].values[0] for t in tasks]
        ci_hi = [model_data[model_data['Task'] == t]['CI_hi'].values[0] for t in tasks]
        # Error bars: distance from mean to CI bounds (always positive)
        errors_lo = [max(0, values[j] - ci_lo[j]) for j in range(len(tasks))]
        errors_hi = [max(0, ci_hi[j] - values[j]) for j in range(len(tasks))]
        offset = (i - (len(plotted_sources) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model,
                     color=COLORS[source],
                     alpha=0.8, yerr=[errors_lo, errors_hi], capsize=3, error_kw={'linewidth': 1})

    # Add reference lines
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect (0.2)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect (0.5)')
    ax.axhline(y=0.8, color='gray', linestyle='-.', alpha=0.5, label='Large effect (0.8)')

    ax.set_xlabel('Task')
    ax.set_ylabel('|Cohen\'s d| (distance from human)')
    ax.set_title('Distributional Distance from Human Responses', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend(loc='upper right', ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 2.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_effect_sizes.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_effect_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig2_effect_sizes.pdf/png")

# ================== FIGURE 3: Semantic Similarity ==================
def create_semantic_clustering_figure():
    """Create a clear visualization showing CRPO's semantic proximity to humans"""

    # Load semantic similarity data
    sim_df = pd.read_csv(ANALYSIS_PATH / "semantic_similarity_matched.csv")

    # Calculate distance from human centroid (1 - similarity)
    # use whichever model columns the semantic-similarity analysis produced
    models = [s for s in LABELS if s != 'human' and f'{s}_centroid_sim' in sim_df.columns]
    model_labels = [LABELS[s] for s in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.3)  # Add spacing between panels

    # Panel A: Centroid similarity by task
    ax = axes[0]
    tasks = sim_df['task'].tolist()
    x = np.arange(len(tasks))
    width = 0.25

    width = 0.8 / max(len(models), 1)
    for i, (model, label) in enumerate(zip(models, model_labels)):
        col = f'{model}_centroid_sim'
        values = sim_df[col].tolist()
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=label, color=COLORS[model], alpha=0.8)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Task')
    ax.set_ylabel('Cosine Similarity to Human Centroid')
    ax.set_title('(A) Semantic Similarity to Humans', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('SCTT-', '') for t in tasks], rotation=20, ha='right')
    ax.legend(loc='lower left', fontsize=8)
    ax.set_ylim(0.95, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Average across tasks
    ax = axes[1]
    avg_sims = []
    for model in models:
        col = f'{model}_centroid_sim'
        avg_sims.append(sim_df[col].mean())

    colors = [COLORS[m] for m in models]
    bars = ax.bar(model_labels, avg_sims, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, val in zip(bars, avg_sims):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('(B) Average Across Tasks', fontweight='bold')
    ax.set_ylim(0.96, 0.99)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_semantic_similarity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_semantic_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: fig3_semantic_similarity.pdf/png")

# ================== MAIN ==================
def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    data = load_data()

    print("\nGenerating figures...")
    create_distribution_figure(data)
    create_effect_size_figure(data)
    create_semantic_clustering_figure()

    print(f"\nAll figures saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
