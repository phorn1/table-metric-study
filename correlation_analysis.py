# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Analyse the correlation between automated metrics and human scores."""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau
from pathlib import Path

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
        h = row["human_scores"][0]
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

    # Scale 0-1 metrics to 1-10 and round to integers
    def scale(vals):
        return np.round(np.array(vals) * 10).astype(int).clip(1, 10)

    human_arr = np.array(human_vals)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "TEDS (scaled 1-10)": (scale(teds_vals), human_arr),
        "TEDS struct. (scaled 1-10)": (scale(teds_struct_vals), human_arr),
        "GriTS-Top (scaled 1-10)": (scale(grits_top_vals), human_arr),
        "GriTS-Con (scaled 1-10)": (scale(grits_con_vals), human_arr),
        "SCORE Content Acc. (1-10)": (scale(score_content_vals), human_arr),
        "SCORE Content Acc. Shifted (1-10)": (scale(score_shifted_vals), human_arr),
        "SCORE Index Acc. (1-10)": (scale(score_index_vals), human_arr),
        "SCORE TEDS (1-10)": (scale(score_teds_vals), human_arr),
        "SCORE TEDS corrected (1-10)": (scale(score_teds_corr_vals), human_arr),
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


def plot_bubble(
    name: str,
    metric_vals: np.ndarray,
    human_vals: np.ndarray,
    correlations: dict[str, tuple[float, float]],
    ax: plt.Axes,
):
    # Count occurrences of each (metric, human) pair
    from collections import Counter
    counts = Counter(zip(metric_vals.tolist(), human_vals.tolist()))
    xs = np.array([k[0] for k in counts])
    ys = np.array([k[1] for k in counts])
    sizes = np.array([v for v in counts.values()])

    ax.scatter(xs, ys, s=sizes * 25, alpha=0.5, edgecolors="steelblue", linewidths=0.5)

    # Trend line
    if np.std(metric_vals) > 0:
        z = np.polyfit(metric_vals, human_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(metric_vals.min(), metric_vals.max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=1.5)

    corr_text = "\n".join(f"{k}={v[0]:.3f} (p={v[1]:.3g})" for k, v in correlations.items())
    ax.text(
        0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=7,
        verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Metric Score")
    ax.set_ylabel("Human Score")
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 11))


def main():
    np.random.seed(42)
    rows = load_data(JSONL_PATH)
    print(f"Loaded {len(rows)} rows with human_scores")

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

    for i, (name, (m_vals, h_vals)) in enumerate(metrics.items()):
        plot_bubble(name, m_vals, h_vals, all_correlations[name], axes[i])

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Metric vs Human Score Correlation", fontsize=14, y=1.01)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "correlation_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

    # Bar chart comparing correlations
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    names = list(all_correlations.keys())
    x = np.arange(len(names))
    width = 0.25

    pearson_vals = [all_correlations[n]["Pearson r"][0] for n in names]
    spearman_vals = [all_correlations[n]["Spearman ρ"][0] for n in names]
    kendall_vals = [all_correlations[n]["Kendall τ"][0] for n in names]

    ax2.bar(x - width, pearson_vals, width, label="Pearson r")
    ax2.bar(x, spearman_vals, width, label="Spearman ρ")
    ax2.bar(x + width, kendall_vals, width, label="Kendall τ")

    ax2.set_ylabel("Correlation Coefficient")
    ax2.set_title("Correlation of Metrics with Human Scores")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.legend()
    ax2.axhline(y=0, color="grey", linestyle="-", linewidth=0.5)
    fig2.tight_layout()

    out_path2 = OUTPUT_DIR / "correlation_comparison.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Bar chart saved to {out_path2}")


if __name__ == "__main__":
    main()