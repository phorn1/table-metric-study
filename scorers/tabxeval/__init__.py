"""TabXEval: two-phase LLM-based table evaluation (TabAlign + TabCompare).

Vendored and adapted from https://github.com/CoRAL-ASU/TabXEval (MIT License),
the reference implementation of:

    Pancholi et al., "TabXEval: Why this is a Bad Table? An eXhaustive Rubric
    for Table Evaluation" (Findings of ACL 2025, arXiv:2505.22176)
"""

from .pipeline import ALIGN_PROMPT, COMPARE_PROMPT, evaluate_pair

__all__ = ["ALIGN_PROMPT", "COMPARE_PROMPT", "evaluate_pair"]
