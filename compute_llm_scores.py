#!/usr/bin/env python3
"""Compute LLM-as-a-judge scores for all table pairs in all_tables.jsonl."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

JSONL_PATH = Path(__file__).parent / "all_tables.jsonl"

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


def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def retry(max_retries: int = MAX_RETRIES):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"  Attempt {attempt + 1} failed: {e}. Retrying...")
            raise last_error
        return wrapper
    return decorator


@retry()
def evaluate_table(client: OpenAI, gt_table: str, extracted_table: str) -> TableEvaluation:
    prompt = TABLE_EVALUATION_PROMPT.format(gt_table=gt_table, extracted_table=extracted_table)
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


def load_entries() -> list[dict]:
    entries = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_entries(entries: list[dict]) -> None:
    with open(JSONL_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    entries = load_entries()
    print(f"Loaded {len(entries)} entries from {JSONL_PATH}")

    # Find entries that still need scoring for this model
    todo = [
        (i, entry)
        for i, entry in enumerate(entries)
        if not any(s["judge_model"] == MODEL for s in entry.get("llm_scores", []))
    ]
    print(f"{len(todo)} entries need evaluation with {MODEL}")
    if not todo:
        return

    client = get_client()
    lock = threading.Lock()
    done = 0

    def process(idx: int, entry: dict) -> None:
        nonlocal done
        result = evaluate_table(client, entry["gt_table"], entry["extracted_table"])
        score_entry = {
            "judge_model": MODEL,
            "score": max(0, min(10, result.score)),
            "errors": result.errors,
        }
        with lock:
            entries[idx].setdefault("llm_scores", []).append(score_entry)
            save_entries(entries)
            done += 1
            print(f"  [{done}/{len(todo)}] {entry['id']}: score={score_entry['score']}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, idx, entry): idx for idx, entry in todo}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                idx = futures[future]
                print(f"  FAILED {entries[idx]['id']}: {exc}")

    print(f"Done. {done}/{len(todo)} entries scored.")


if __name__ == "__main__":
    main()