# Table Metric Study

Meta-evaluation of table extraction metrics against human judgment, accompanying the paper:

> **Benchmarking PDF Parsers on Table Extraction with LLM-based Semantic Evaluation**

This repository provides implementations of rule-based table metrics (TEDS, GriTS, SCORE), LLM-as-a-judge scoring, a human evaluation interface, and the correlation analysis used to validate that LLM-based evaluation substantially outperforms rule-based metrics in agreement with human judgment.

## Results

The dataset includes over 1,500 human quality ratings on 518 table pairs. The correlation analysis shows that LLM-based judges achieve substantially higher agreement with human judgment than rule-based metrics:

![Correlation of automated metrics with human scores](correlation_plots/correlation.png)

Correlation of each metric with the averaged human scores (three annotators per table pair):

| Metric                      | Pearson r | Spearman ρ | Kendall τ | Cost / 1k pairs ($) |
| --------------------------- | --------: | ---------: | --------: | ------------------: |
| TEDS                        |     0.684 |      0.717 |     0.558 |                   — |
| TEDS struct.                |     0.627 |      0.720 |     0.579 |                   — |
| GriTS-Top                   |     0.633 |      0.735 |     0.597 |                   — |
| GriTS-Con                   |     0.701 |      0.745 |     0.598 |                   — |
| GriTS-Avg                   |     0.698 |      0.765 |     0.606 |                   — |
| SCORE Index                 |     0.558 |      0.684 |     0.561 |                   — |
| SCORE Content               |     0.642 |      0.657 |     0.524 |                   — |
| SCORE Content Shifted       |     0.648 |      0.653 |     0.526 |                   — |
| SCORE-Avg                   |     0.637 |      0.687 |     0.541 |                   — |
| LLM: claude-opus-4.6        |     0.939 |      0.891 |     0.804 |                7.60 |
| LLM: gemma-4-31b-it         |     0.929 |      0.884 |     0.796 |                0.18 |
| LLM: gemini-3-flash-preview |     0.924 |      0.892 |     0.803 |                0.78 |
| LLM: gemma-4-26b-a4b-it     |     0.909 |      0.861 |     0.766 |                0.54 |
| LLM: gemini-3.1-flash-lite-preview |     0.909 |      0.851 |     0.754 |                0.36 |
| LLM: gpt-5.4-nano           |     0.809 |      0.799 |     0.683 |                0.28 |
| LLM: deepseek-v3.2          |     0.780 |      0.805 |     0.699 |                0.42 |
| LLM: mistral-small-2603     |     0.756 |      0.799 |     0.685 |                0.28 |

LLM judge costs are based on OpenRouter pricing as of 2026-04-24.

## Project Structure

| File                      | Description                                                                                    |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| `all_tables.json`         | Central dataset: ground truth tables, parser extractions, all metric scores, and human ratings |
| `compute_metrics.py`      | Compute rule-based metrics (TEDS, GriTS, SCORE) for all extractions                            |
| `compute_llm_scores.py`   | LLM-as-a-judge scoring via OpenRouter API                                                      |
| `latex_to_html_claude.py` | Convert LaTeX ground truth tables to HTML (required by rule-based metrics)                     |
| `human_eval.py`           | Gradio web UI for human annotation (0–10 scoring)                                              |
| `correlation_analysis.py` | Correlation analysis and scatter plots (generates paper figures)                               |
| `scorers/`                | Metric implementations (TEDS, GriTS, SCORE, table normalization)                               |

## Reproducing

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). All scripts can be run via `uv run python <script>.py`.

```bash
uv sync
```

System dependencies for rule-based metrics and human evaluation UI:
- `pdflatex` and `pdftoppm` (e.g., via TeX Live)
- `latexmlc` (for LaTeX-to-HTML normalization)

LLM scoring requires an OpenRouter API key (`export OPENROUTER_API_KEY=...`).

## Data Format

Each entry in `all_tables.json` pairs a ground truth table with its parser extractions, metric scores, and human ratings:

```json
{
  "gt_id": "000_00",
  "gt_table": "\\begin{tabular}...",
  "gt_table_html": "<table>...</table>",
  "complexity": "simple | moderate | complex",
  "extractions": [
    {
      "parser": "gemini_3_flash",
      "extracted_table": "...",
      "metrics": { "teds": 0.91, "grits_top": 0.89, "grits_con": 0.87, ... },
      "llm_scores": [
        { "judge_model": "google/gemini-3-flash-preview", "score": 9, "errors": [...] }
      ],
      "human_scores": [8, 8, 7]
    }
  ]
