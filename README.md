# Table Metric Study

Meta-evaluation of table extraction metrics against human judgment, accompanying the paper:

> **Beyond String Matching: Semantic Evaluation of PDF Table Extraction**

This repository provides implementations of rule-based table metrics (TEDS, GriTS, SCORE), LLM-as-a-judge scoring, a human evaluation interface, and the correlation analysis used to validate that LLM-based evaluation substantially outperforms rule-based metrics in agreement with human judgment.

## Results

The dataset includes over 1,500 human quality ratings on 518 table pairs. The correlation analysis shows that LLM-based judges achieve substantially higher agreement with human judgment than rule-based metrics:

![Correlation of automated metrics with human scores](correlation_plots/correlation.png)

Correlation of each metric with the averaged human scores (three annotators per table pair):

| Metric                             | Pearson r | Spearman ρ | Kendall τ | Cost / 1k pairs ($) |
| ---------------------------------- | --------: | ---------: | --------: | ------------------: |
| TEDS                               |     0.684 |      0.717 |     0.558 |                   — |
| TEDS struct.                       |     0.627 |      0.720 |     0.579 |                   — |
| GriTS-Top                          |     0.633 |      0.735 |     0.597 |                   — |
| GriTS-Con                          |     0.701 |      0.745 |     0.598 |                   — |
| GriTS-Avg                          |     0.698 |      0.765 |     0.606 |                   — |
| SCORE Index                        |     0.558 |      0.684 |     0.561 |                   — |
| SCORE Content                      |     0.642 |      0.657 |     0.524 |                   — |
| SCORE Content Shifted              |     0.648 |      0.653 |     0.526 |                   — |
| SCORE-Avg                          |     0.637 |      0.687 |     0.541 |                   — |
| TabXEval (gpt-4o)                  |     0.570 |      0.618 |     0.515 |               28.19 |
| LLM: claude-opus-4.6               |     0.939 |      0.891 |     0.804 |                7.60 |
| LLM: gemma-4-31b-it                |     0.929 |      0.884 |     0.796 |                0.18 |
| LLM: gemini-3-flash-preview        |     0.924 |      0.892 |     0.803 |                0.78 |
| LLM: gemma-4-26b-a4b-it            |     0.909 |      0.861 |     0.766 |                0.54 |
| LLM: gemini-3.1-flash-lite-preview |     0.909 |      0.851 |     0.754 |                0.36 |
| LLM: gpt-5.4-nano                  |     0.809 |      0.799 |     0.683 |                0.28 |
| LLM: deepseek-v3.2                 |     0.780 |      0.805 |     0.699 |                0.42 |
| LLM: mistral-small-2603            |     0.756 |      0.799 |     0.685 |                0.28 |

LLM judge costs are based on OpenRouter pricing as of 2026-04-24. The TabXEval cost was measured on a 5-pair sample (two gpt-4o calls per pair) at OpenRouter pricing as of 2026-07-04.

Notably, [TabXEval](https://github.com/CoRAL-ASU/TabXEval) — a two-phase LLM-based evaluation pipeline (alignment + fine-grained comparison, run here with its original prompts and gpt-4o judge) — correlates *worse* with human judgment than the rule-based metrics on this benchmark, despite costing more than any single-prompt LLM judge. Its penalty-based score saturates at the top: 311 of 518 pairs (60%) receive exactly 1.0, including extractions that humans rated as low as 0.7/10. Two caveats apply: the pipeline inherits the reference implementation's sampling parameters (temperature 0.1), so scores are not fully deterministic across runs, and TabXEval was designed for general table comparison rather than PDF-extraction evaluation specifically.

### Prompt sensitivity

Same 518 extractions, three prompt variants:

**`tuned`** — the prompt used for the main results. Response schema `{errors[], score}`.

```
You are a strict table evaluator. Your task is to determine if the extracted table correctly represents the ground truth table, focusing on content accuracy, structural preservation, and information completeness. The extracted table was parsed from the rendered table. Disregard LaTeX-specific elements in the ground truth (e.g., comments, styling commands, font formatting) that have no effect on content or structure.

Ground Truth Table (LaTeX):
{gt_table}

Extracted Table:
{extracted_table}

Evaluate the extracted table using the following criteria:
1. Content accuracy: Are all cell values, headers, and data correctly preserved?
2. Structure preservation: Are all rows and columns present, and can each value be unambiguously mapped to its row/column headers? Broken or ambiguous associations count as errors.

Note: Different output formats (markdown, HTML, plain text) are acceptable as long as no information is lost. Apply this key test: Could a reader who sees ONLY the extracted table — without access to the ground truth — unambiguously reconstruct every cell-to-header mapping and all content of the original table? If not, consider the parsing as failed and assign a low score.

First, enumerate up to 5 of the most significant errors and ambiguities found. Then assign a score from 0 to 10, where 10 is a perfect match.
```

**`tuned_no_cot`** — identical to `tuned` with the final sentence replaced by *"Assign a score from 0 to 10, where 10 is a perfect match."* and response schema reduced to `{score}`. Isolates the CoT scaffold.

**`naive`** — `"Rate how well the extracted table matches the ground truth table on a scale from 0 to 10."` Response schema `{score}`.

Both deltas are taken relative to the naive baseline: Δ content = tuned_no_cot − naive, Δ full = tuned − naive.

| Model                         | Pearson (tuned) | Pearson (tuned_no_cot) | Pearson (naive) | Δ content | Δ full |
| ----------------------------- | --------------: | ---------------------: | --------------: | --------: | -----: |
| gemini-3.1-flash-lite-preview |           0.909 |                  0.863 |           0.807 |    +0.056 | +0.102 |
| gpt-5.4-nano                  |           0.809 |                  0.795 |           0.718 |    +0.076 | +0.091 |
| gemini-3-flash-preview        |           0.924 |                  0.907 |           0.851 |    +0.056 | +0.073 |
| gemma-4-26b-a4b-it            |           0.909 |                  0.777 |           0.856 |    −0.078 | +0.054 |
| claude-opus-4.6               |           0.939 |                  0.933 |           0.891 |    +0.042 | +0.048 |
| gemma-4-31b-it                |           0.929 |                  0.916 |           0.907 |    +0.009 | +0.022 |
| deepseek-v3.2                 |           0.780 |                  0.671 |           0.779 |    −0.108 | +0.001 |
| mistral-small-2603            |           0.756 |                  0.842 |           0.803 |    +0.040 | −0.046 |

The CoT scaffold and engineered content interact: gemma-4-26b and deepseek-v3.2 show negative Δ content but positive Δ full, while mistral-small shows the inverse. Of 24 (model × variant) cells, only deepseek-v3.2/tuned_no_cot (0.671) falls below the best rule-based metric (GriTS-Con 0.701).

## Project Structure

| File                      | Description                                                                                    |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| `all_tables.json`         | Central dataset: ground truth tables, parser extractions, all metric scores, and human ratings |
| `compute_metrics.py`      | Compute rule-based metrics (TEDS, GriTS, SCORE) for all extractions                            |
| `compute_llm_scores.py`   | LLM-as-a-judge scoring via OpenRouter API                                                      |
| `compute_tabxeval_scores.py` | TabXEval scoring (two-phase LLM alignment + comparison) via OpenRouter API                  |
| `latex_to_html_claude.py` | Convert LaTeX ground truth tables to HTML (required by rule-based metrics)                     |
| `human_eval.py`           | Gradio web UI for human annotation (0–10 scoring)                                              |
| `correlation_analysis.py` | Correlation analysis and scatter plots (generates paper figures)                               |
| `scorers/`                | Metric implementations (TEDS, GriTS, SCORE, table normalization)                               |
| `scorers/tabxeval/`       | [TabXEval](https://github.com/CoRAL-ASU/TabXEval) pipeline (vendored, MIT) with original prompts and ported notebook scoring |

## Reproducing

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). All scripts can be run via `uv run python <script>.py`.

```bash
uv sync
```

System dependencies for rule-based metrics and human evaluation UI:
- `pdflatex` and `pdftoppm` (e.g., via TeX Live)
- `latexmlc` (for LaTeX-to-HTML normalization)

LLM scoring requires an OpenRouter API key (`export OPENROUTER_API_KEY=...`).

TabXEval ([Pancholi et al., Findings of ACL 2025](https://arxiv.org/abs/2505.22176)) is included as an additional LLM-based baseline via `compute_tabxeval_scores.py`. It aligns ground truth and extraction with a rule-based fuzzy merge plus an LLM alignment call (TabAlign), classifies every non-exact cell difference with a second LLM call (TabCompare), and aggregates the classified differences into a penalty-based score in [0, 1]. The implementation under `scorers/tabxeval/` is vendored from the [official repository](https://github.com/CoRAL-ASU/TabXEval) (MIT License) with the original prompts and gpt-4o judge (via OpenRouter), and the scoring logic ported from its `eval_scores.ipynb`.

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
      "tabxeval": { "judge_model": "openai/gpt-4o", "score": 0.86 },
      "human_scores": [8, 8, 7]
    }
  ]
}
```
