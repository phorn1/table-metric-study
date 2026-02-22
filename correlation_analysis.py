# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "seaborn",
# ]
# ///
"""Analyse the correlation between automated metrics and human scores."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10

JSONL_PATH = Path(__file__).parent / "all_tables.jsonl"
OUTPUT_DIR = Path(__file__).parent / "correlation_plots"


def load_data(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if "human_scores" in obj and obj.get("extracted_table", "").strip():
                rows.append(obj)
    return rows


def extract_metric_vectors(rows: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {metric_name: (metric_values, human_values)} for all metrics."""
    teds_vals, teds_struct_vals, human_vals = [], [], []
    grits_top_vals, grits_con_vals = [], []
    score_content_vals, score_shifted_vals, score_index_vals = [], [], []
    score_teds_vals, score_teds_corr_vals = [], []
    llm_scores: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))

    for row in rows:
        h = np.mean(row["human_scores"])
        human_vals.append(h)
        teds_vals.append(row["teds"])
        teds_struct_vals.append(row["teds_structure_only"])
        grits_top_vals.append(row.get("grits_top", 0.0))
        grits_con_vals.append(row.get("grits_con", 0.0))
        score_content_vals.append(row.get("score_cell_content_acc", 0.0))
        score_shifted_vals.append(row.get("score_shifted_content_acc", 0.0))
        score_index_vals.append(row.get("score_cell_index_acc", 0.0))
        score_teds_vals.append(row.get("score_teds", 0.0))
        score_teds_corr_vals.append(row.get("score_teds_corrected", 0.0))

        for entry in row.get("llm_scores", []):
            model = entry["judge_model"]
            llm_scores[model][0].append(entry["score"])
            llm_scores[model][1].append(h)

    def scale(vals):
        """Scale 0-1 metrics to 0-10."""
        return np.array(vals) * 10

    human_arr = np.array(human_vals)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "TEDS": (scale(teds_vals), human_arr),
        "TEDS struct.": (scale(teds_struct_vals), human_arr),
        "GriTS-Top": (scale(grits_top_vals), human_arr),
        "GriTS-Con": (scale(grits_con_vals), human_arr),
        "GriTS-Avg": (scale((np.array(grits_top_vals) + np.array(grits_con_vals)) / 2), human_arr),
        "SCORE Index": (scale(score_index_vals), human_arr),
        "SCORE Content": (scale(score_content_vals), human_arr),
        "SCORE Content Shifted": (scale(score_shifted_vals), human_arr),
        "SCORE-Avg": (scale((np.array(score_content_vals) + np.array(score_index_vals)) / 2), human_arr),
        "SCORE TEDS": (scale(score_teds_vals), human_arr),
        "SCORE TEDS corrected": (scale(score_teds_corr_vals), human_arr),
    }
    for model, (scores, humans) in sorted(llm_scores.items()):
        label = model.split("/")[-1] if "/" in model else model
        result[f"LLM: {label}"] = (np.array(scores), np.array(humans))

    return result


def compute_correlations(
    metric_vals: np.ndarray, human_vals: np.ndarray
) -> dict[str, tuple[float, float]]:
    pearson_r, pearson_p = pearsonr(metric_vals, human_vals)
    spearman_r, spearman_p = spearmanr(metric_vals, human_vals)
    kendall_t, kendall_p = kendalltau(metric_vals, human_vals)
    return {
        "Pearson r": (pearson_r, pearson_p),
        "Spearman ρ": (spearman_r, spearman_p),
        "Kendall τ": (kendall_t, kendall_p),
    }


def plot_scatter(ax, title, metric_vals, human_vals, color, metrics_text,
                 small=False):
    """Plot a complete bubble scatter subplot, matching cdm_vs_llm_plotter style."""
    plot_metrics = np.round(metric_vals).astype(int).clip(0, 10)
    plot_humans = np.round(human_vals).astype(int).clip(0, 10)
    counts = Counter(zip(plot_humans.tolist(), plot_metrics.tolist()))

    # Plot points with size coding
    for (x, y), count in counts.items():
        size = 50 + count * 30
        alpha = min(0.8, 0.4 + count * 0.1)
        ax.scatter(x, y, s=size, alpha=alpha, color=color,
                   edgecolors='white', linewidth=1.5)
        if count > 1:
            ax.annotate(str(count), (x, y), ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')

    # Perfect agreement line
    ax.plot([0, 10], [0, 10], 'r--', alpha=0.8, linewidth=2, label='Perfect Agreement')
    ax.set_xlabel('Human Scores (avg)')
    ax.set_ylabel(title)
    ax.set_title(f'{title} vs Human Scores')
    ax.grid(True, alpha=0.3)

    # Position legend inside plot at bottom center with better visibility
    fs = 8 if small else 10
    legend = ax.legend(loc='lower center', frameon=True, fancybox=True,
                       shadow=True, fontsize=fs)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(range(0, 11))
    ax.set_yticks(range(0, 11))
    ax.set_aspect('equal')

    # Add metrics text in bottom right corner with better visibility
    ax.text(0.915, 0.05, metrics_text, transform=ax.transAxes,
            fontsize=fs, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                      edgecolor='black', linewidth=1))


def main():
    np.random.seed(42)
    rows = load_data(JSONL_PATH)
    print(f"Loaded {len(rows)} rows with human_scores")

    # Inter-annotator agreement (rows with >= 2 human scores)
    multi = [r["human_scores"] for r in rows if len(r["human_scores"]) >= 2]
    if multi:
        h1 = np.array([s[0] for s in multi])
        h2 = np.array([s[1] for s in multi])
        iaa_pearson, _ = pearsonr(h1, h2)
        iaa_spearman, _ = spearmanr(h1, h2)
        iaa_kendall, _ = kendalltau(h1, h2)
        mean_abs_diff = np.mean(np.abs(h1 - h2))
        print(f"\nInter-annotator agreement ({len(multi)} rows with ≥2 scores):")
        print(f"  Pearson r:  {iaa_pearson:.4f}")
        print(f"  Spearman ρ: {iaa_spearman:.4f}")
        print(f"  Kendall τ:  {iaa_kendall:.4f}")
        print(f"  Mean |diff|: {mean_abs_diff:.2f}")

    metrics = extract_metric_vectors(rows)

    # Print correlation table
    print(f"\n{'Metric':<30} {'Pearson r':>10} {'Spearman ρ':>11} {'Kendall τ':>10}  {'n':>4}")
    print("-" * 72)
    all_correlations = {}
    for name, (m_vals, h_vals) in metrics.items():
        corrs = compute_correlations(m_vals, h_vals)
        all_correlations[name] = corrs
        print(
            f"{name:<30} {corrs['Pearson r'][0]:>10.4f} "
            f"{corrs['Spearman ρ'][0]:>11.4f} {corrs['Kendall τ'][0]:>10.4f}  {len(m_vals):>4}"
        )

    # Plot
    n_metrics = len(metrics)
    cols = 3
    rows_grid = (n_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows_grid, cols, figsize=(5 * cols, 4 * rows_grid))
    axes = axes.flatten()

    palette = sns.color_palette("tab10")
    for i, (name, (m_vals, h_vals)) in enumerate(metrics.items()):
        corrs = all_correlations[name]
        metrics_text = (f'Pearson r: {corrs["Pearson r"][0]:.3f}\n'
                        f'Spearman \u03c1: {corrs["Spearman ρ"][0]:.3f}\n'
                        f'Kendall \u03c4: {corrs["Kendall τ"][0]:.3f}')
        plot_scatter(axes[i], name, m_vals, h_vals,
                     palette[i % len(palette)], metrics_text)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Metric vs Human Score Correlation", fontsize=14, y=1.01)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "correlation_all.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

    # --- 3x2 paper figure (3 rows, 2 cols) with selected metrics ---
    paper_metrics = [
        "TEDS",                        "LLM: deepseek-v3.2",
        "GriTS-Avg",                   "LLM: gemini-3-flash-preview",
        "SCORE-Avg",                   "LLM: claude-opus-4.6",
    ]
    paper_labels = [
        "TEDS",                        "DeepSeek-v3.2",
        "GriTS-Avg",                   "Gemini-3-Flash",
        "SCORE-Avg",                   "Claude Opus 4.6",
    ]
    paper_row_labels = [
        "Rule-based",  "LLM-as-a-Judge",
        "Rule-based",  "LLM-as-a-Judge",
        "Rule-based",  "LLM-as-a-Judge",
    ]
    paper_colors = [
        sns.color_palette("tab10")[0],  # blue
        sns.color_palette("tab10")[1],  # orange
        sns.color_palette("tab10")[2],  # green
        sns.color_palette("tab10")[3],  # red
        sns.color_palette("tab10")[4],  # purple
        sns.color_palette("tab10")[6],  # pink
    ]

    fig3, axes3 = plt.subplots(3, 2, figsize=(10, 5 * 3),
                               constrained_layout=True)
    axes3_flat = axes3.flatten()

    for idx, (metric_key, label, row_label, color) in enumerate(
        zip(paper_metrics, paper_labels, paper_row_labels, paper_colors)
    ):
        ax = axes3_flat[idx]
        m_vals, h_vals = metrics[metric_key]
        corrs = all_correlations[metric_key]

        metrics_text = (f'Corr: {corrs["Pearson r"][0]:.3f}\n'
                        f'Spearman: {corrs["Spearman ρ"][0]:.3f}\n'
                        f'Kendall: {corrs["Kendall τ"][0]:.3f}')
        plot_scatter(ax, f'{label} ({row_label})', m_vals, h_vals,
                     color, metrics_text, small=True)

    out_path3 = OUTPUT_DIR / "correlation.pdf"
    fig3.savefig(out_path3, dpi=300, bbox_inches="tight")
    out_path3_png = OUTPUT_DIR / "correlation.png"
    fig3.savefig(out_path3_png, dpi=300, bbox_inches="tight")
    print(f"Paper figure saved to {out_path3} and {out_path3_png}")


if __name__ == "__main__":
    main()