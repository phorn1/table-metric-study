#!/usr/bin/env python3
"""Compute TabXEval scores for all table pairs in all_tables.json.

TabXEval (https://github.com/CoRAL-ASU/TabXEval) is a two-phase LLM-based
table evaluation: an alignment call (TabAlign) merges ground truth and
extraction into one cell-level aligned table, a comparison call (TabCompare)
classifies every non-exact cell difference, and a rule-based penalty
aggregation turns the classified differences into a score in [0, 1].

Both tables are canonicalized to markdown grids (via the same HTML
normalization used for the rule-based metrics) before entering the pipeline.
Results are stored per extraction under ext["tabxeval"]; extractions that
already have an entry are skipped, so the script can be re-run to fill gaps.
"""

import argparse
import json
import os
import threading
import time
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

from scorers.normalize import normalize_table
from scorers.tabxeval import evaluate_pair

load_dotenv()

DATA_PATH = Path(__file__).parent / "all_tables.json"

# gpt-4o is the judge hardcoded in the reference implementation; temperature
# and top_p match evaluation_pipeline/eval.py.
MODEL = "openai/gpt-4o"
TEMPERATURE = 0.1
TOP_P = 0.2

MAX_WORKERS = 8
MAX_RETRIES = 10


def html_table_to_markdown(html_str: str) -> str | None:
    """Convert an HTML table to a markdown pipe table (TabXEval's input format).

    Rowspans/colspans are expanded by duplicating the cell value. Pipe
    characters inside cells are replaced (TabXEval splits rows on '|'),
    and empty cells become '-' so that columns keep their position.
    """
    soup = BeautifulSoup(html_str, "html.parser")
    table = soup.find("table")
    if table is None:
        return None

    grid: dict[tuple[int, int], str] = {}
    for row_idx, tr in enumerate(table.find_all("tr")):
        col_idx = 0
        for cell in tr.find_all(["td", "th"]):
            while (row_idx, col_idx) in grid:
                col_idx += 1
            try:
                rowspan = int(cell.get("rowspan") or 1)
                colspan = int(cell.get("colspan") or 1)
            except ValueError:
                rowspan = colspan = 1
            text = " ".join(cell.get_text(" ", strip=True).split())
            text = text.replace("|", "¦") or "-"
            for r in range(rowspan):
                for c in range(colspan):
                    grid[(row_idx + r, col_idx + c)] = text
            col_idx += colspan

    if not grid:
        return None
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    rows = [[grid.get((r, c), "-") for c in range(n_cols)] for r in range(n_rows)]

    lines = ["| " + " | ".join(rows[0]) + " |",
             "| " + " | ".join(["---"] * n_cols) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def to_markdown_table(raw: str) -> str:
    """Canonicalize a table (HTML/markdown/plain text) to a markdown grid.

    Falls back to the raw string if no table structure can be recognized —
    the alignment LLM then works directly on the unparsed extraction.
    """
    try:
        html = normalize_table(raw)
        md = html_table_to_markdown(html) if html else None
    except Exception:
        md = None
    return md if md else raw


def make_ask_llm(client: OpenAI):
    def ask_llm(system_prompt: str, user_text: str) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response")
                return content
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"  Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2)
                else:
                    raise
        raise RuntimeError("unreachable")

    return ask_llm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N pending extractions (pilot run)")
    args = parser.parse_args()

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    total = sum(len(gt["extractions"]) for gt in data)
    print(f"Loaded {len(data)} GT tables, {total} extractions from {DATA_PATH}")

    todo = []
    for gt in data:
        for ext in gt["extractions"]:
            if "tabxeval" in ext:
                continue
            if not ext["extracted_table"].strip():
                continue
            todo.append((gt, ext))
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"{len(todo)} extractions need TabXEval scoring with judge={MODEL}")
    if not todo:
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required.")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=0,
        timeout=180,
    )
    ask_llm = make_ask_llm(client)

    gt_markdown = {gt["gt_id"]: html_table_to_markdown(gt["gt_table_html"]) for gt in data}

    lock = threading.Lock()
    done = 0

    def process(gt: dict, ext: dict) -> None:
        nonlocal done
        gt_md = gt_markdown[gt["gt_id"]]
        if gt_md is None:
            raise ValueError(f"GT table {gt['gt_id']} could not be converted to markdown")
        ext_md = to_markdown_table(ext["extracted_table"])
        result = evaluate_pair(gt_md, ext_md, ask_llm)
        entry = {"judge_model": MODEL, "score": round(result["score"], 4)}
        ext_id = f"{ext['parser']}_{gt['gt_id']}"
        with lock:
            ext["tabxeval"] = entry
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            done += 1
            print(f"  [{done}/{len(todo)}] {ext_id}: tabxeval={entry['score']:.4f}")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, gt, ext): (gt, ext) for gt, ext in todo}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                gt, ext = futures[future]
                print(f"  FAILED {ext['parser']}_{gt['gt_id']}: {exc!r}")

    print(f"Done. {done}/{len(todo)} pairs scored.")


if __name__ == "__main__":
    main()
