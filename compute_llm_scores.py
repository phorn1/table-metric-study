#!/usr/bin/env python3
"""Compute LLM-as-a-judge scores for all table pairs in all_tables.json."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

DATA_PATH = Path(__file__).parent / "all_tables.json"

MODEL = "openai/gpt-5-mini"
MAX_WORKERS = 8
MAX_RETRIES = 10

TABLE_EVALUATION_PROMPT = """\
You are a strict table evaluator. Your task is to determine if the extracted table correctly represents the ground truth table, focusing on content accuracy, structural preservation, and information completeness. The extracted table was parsed from the rendered table. Disregard LaTeX-specific elements in the ground truth (e.g., comments, styling commands, font formatting) that have no effect on content or structure.

Ground Truth Table (LaTeX):
{gt_table}

Extracted Table:
{extracted_table}

Evaluate the extracted table using the following criteria:
1. Content accuracy: Are all cell values, headers, and data correctly preserved?
2. Structure preservation: Are all rows and columns present, and can each value be unambiguously mapped to its row/column headers? Broken or ambiguous associations count as errors.

Note: Different output formats (markdown, HTML, plain text) are acceptable as long as no information is lost. Apply this key test: Could a reader who sees ONLY the extracted table — without access to the ground truth — unambiguously reconstruct every cell-to-header mapping and all content of the original table? If not, consider the parsing as failed and assign a low score.

First, enumerate all errors and ambiguities found. Then assign a score from 0 to 10, where 10 is a perfect match.\
"""


class TableEvaluation(BaseModel):
    errors: list[str]
    score: int


def evaluate_table(client: OpenAI, gt_table: str, extracted_table: str) -> TableEvaluation:
    prompt = TABLE_EVALUATION_PROMPT.format(gt_table=gt_table, extracted_table=extracted_table)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "TableEvaluation",
                        "strict": True,
                        "schema": TableEvaluation.model_json_schema(),
                    },
                },
            )
            return TableEvaluation.model_validate_json(response.choices[0].message.content)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying...")
            else:
                raise


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    total = sum(len(gt["extractions"]) for gt in data)
    print(f"Loaded {len(data)} GT tables, {total} extractions from {DATA_PATH}")

    # Find extractions that still need scoring for this model
    todo = []
    for gt in data:
        for ext in gt["extractions"]:
            if MODEL not in {s["judge_model"] for s in ext["llm_scores"]}:
                todo.append((gt, ext))

    print(f"{len(todo)} extractions need evaluation with {MODEL}")
    if not todo:
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    lock = threading.Lock()
    done = 0

    def process(gt: dict, ext: dict) -> None:
        nonlocal done
        result = evaluate_table(client, gt["gt_table"], ext["extracted_table"])
        score_entry = {
            "judge_model": MODEL,
            "score": max(0, min(10, result.score)),
            "errors": result.errors,
        }
        ext_id = f"{ext['parser']}_{gt['gt_id']}"
        with lock:
            ext["llm_scores"].append(score_entry)
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            done += 1
            print(f"  [{done}/{len(todo)}] {ext_id}: score={score_entry['score']}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, gt, ext): (gt, ext) for gt, ext in todo}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                gt, ext = futures[future]
                print(f"  FAILED {ext['parser']}_{gt['gt_id']}: {exc}")

    print(f"Done. {done}/{len(todo)} extractions scored.")


if __name__ == "__main__":
    main()