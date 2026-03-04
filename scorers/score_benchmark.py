"""Adapter for the SCORE benchmark table metrics.

Wraps the eval-metrics-core package from:
https://github.com/Unstructured-IO/unstructured-eval-metrics
"""

from scoring.table_scoring import score_tables

ZERO_SCORES = {
    "score_content": 0.0,
    "score_content_shifted": 0.0,
    "score_index": 0.0,
}


def score_from_html(gt_html: str, pred_html: str) -> dict:
    """Compute SCORE benchmark metrics for a single GT/pred HTML table pair."""
    if not gt_html or not pred_html:
        return dict(ZERO_SCORES)

    gt_element = [{"type": "Table", "metadata": {"text_as_html": gt_html}}]
    pred_element = [{"type": "Table", "metadata": {"text_as_html": pred_html}}]

    try:
        result = score_tables(
            sample_table=pred_element,
            ground_truth=gt_element,
            sample_format="html",
            ground_truth_format="html",
        )
    except Exception:
        return dict(ZERO_SCORES)

    s = result.scores
    return {
        "score_content": s.cell_level_content_acc,
        "score_content_shifted": s.shifted_cell_content_acc,
        "score_index": s.cell_level_index_acc,
    }