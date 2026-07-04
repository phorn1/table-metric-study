"""TabXEval two-phase evaluation pipeline (TabAlign + TabCompare).

Port of the per-pair loop in TabXEval's evaluation_pipeline/eval.py
(https://github.com/CoRAL-ASU/TabXEval, MIT License), decoupled from the
OpenAI client: the caller supplies `ask_llm(system_prompt, user_text) -> str`.

Phase 1 (TabAlign): a rule-based fuzzy merge produces a partial alignment,
which the LLM completes into a full cell-level alignment (`t1_value/t2_value`).
Phase 2 (TabCompare): cells that do not match exactly are sent to the LLM,
which classifies each difference by data type, entity, and error category.
The classified differences are aggregated into the penalty defined in
scoring.py; the reported score is `1 - penalty` (1.0 = perfect match).

Deviation from upstream: the TabCompare call is skipped when every cell
matches exactly (upstream still calls the LLM, but the notebook scoring
short-circuits this case to a zero penalty, so the call result is unused).
"""

from pathlib import Path

from .comparison import (
    compare,
    get_partial_cells_stats,
    make_delta_stats_table,
    parse_string,
    table_to_dict_list_comparison,
)
from .fuzzy_matching import merge_tables_fuzzy
from .scoring import ALLOWED_DATA_TYPES, compute_penalty

_PROMPT_DIR = Path(__file__).parent
ALIGN_PROMPT = (_PROMPT_DIR / "align_prompt.txt").read_text(encoding="utf-8").strip()
COMPARE_PROMPT = (_PROMPT_DIR / "compare_prompt.txt").read_text(encoding="utf-8").strip()


def evaluate_pair(table1: str, table2: str, ask_llm) -> dict:
    """Run the full TabXEval pipeline on one (reference, prediction) pair.

    Returns {"score": float in [0, 1], "penalty": float,
             "alignment": str, "comparison_tuples": str}.
    Raises on unrecoverable failures (e.g. the LLM alignment is not a table).
    """
    partial_alignment = merge_tables_fuzzy(table1, table2)[0]

    user_text = f"Align the following tables:\n\n{table1}\n\n{table2}\n\n"
    if partial_alignment is not None:
        user_text += f"Partially Aligned Table:{partial_alignment}"
    alignment = ask_llm(ALIGN_PROMPT, user_text)

    df_replaced, df_compare, df_wo_em = compare(alignment)

    if df_wo_em.empty:
        comparison_tuples = ""
    else:
        comparison_tuples = ask_llm(COMPARE_PROMPT, df_wo_em.to_markdown(index=False))

    tuples_list = table_to_dict_list_comparison(comparison_tuples) if comparison_tuples else []
    comparison_tuples_parsed = []
    for row in tuples_list:
        row_updated = {}
        for key, value in row.items():
            row_updated[key] = None if value is None else parse_string(value)
        comparison_tuples_parsed.append(row_updated)

    record = {
        "table1": table1,
        "table2": table2,
        "partial_alignment": partial_alignment,
        "alignment": alignment,
        "df_replaced": df_replaced,
        "df_compare": df_compare,
        "df_wo_em": df_wo_em,
        "comparison_tuples": comparison_tuples,
        "comparison_tuples_parsed": comparison_tuples_parsed,
    }
    record = get_partial_cells_stats([record], ALLOWED_DATA_TYPES)[0]
    record["partial_cell_delta_stats"] = make_delta_stats_table(
        record["delta"], record["type_counts"])

    penalty = compute_penalty(record)
    return {
        "score": 1 - penalty,
        "penalty": penalty,
        "alignment": alignment,
        "comparison_tuples": comparison_tuples,
    }
