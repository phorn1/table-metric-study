#!/usr/bin/env python3
"""Compute LLM-as-a-judge scores for all table pairs in all_tables.json.

Three prompt variants are supported:
- "tuned": engineered prompt with criteria, role, and CoT-style error enumeration
- "tuned_no_cot": tuned prompt content but no error-enumeration scaffold (ablation)
- "naive": minimal "rate from 0 to 10" prompt with no scaffolding

Scores are keyed by (judge_model, prompt_variant) — set PROMPT_VARIANT and
MODELS below, then run.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

DATA_PATH = Path(__file__).parent / "all_tables.json"

MODELS = [
    "anthropic/claude-opus-4.6",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemma-4-31b-it",
    "google/gemma-4-26b-a4b-it",
    "openai/gpt-5.4-nano",
    "deepseek/deepseek-v3.2",
    "mistralai/mistral-small-2603",
]

PROMPT_VARIANT = "tuned"  # "tuned", "tuned_no_cot", or "naive"

MAX_WORKERS = 8
MAX_RETRIES = 10

TUNED_PROMPT = """\
You are a strict table evaluator. Your task is to determine if the extracted table correctly represents the ground truth table, focusing on content accuracy, structural preservation, and information completeness. The extracted table was parsed from the rendered table. Disregard LaTeX-specific elements in the ground truth (e.g., comments, styling commands, font formatting) that have no effect on content or structure.

Ground Truth Table (LaTeX):
{gt_table}

Extracted Table:
{extracted_table}

Evaluate the extracted table using the following criteria:
1. Content accuracy: Are all cell values, headers, and data correctly preserved?
2. Structure preservation: Are all rows and columns present, and can each value be unambiguously mapped to its row/column headers? Broken or ambiguous associations count as errors.

Note: Different output formats (markdown, HTML, plain text) are acceptable as long as no information is lost. Apply this key test: Could a reader who sees ONLY the extracted table — without access to the ground truth — unambiguously reconstruct every cell-to-header mapping and all content of the original table? If not, consider the parsing as failed and assign a low score.

First, enumerate up to 5 of the most significant errors and ambiguities found. Then assign a score from 0 to 10, where 10 is a perfect match.\
"""

TUNED_NO_COT_PROMPT = """\
You are a strict table evaluator. Your task is to determine if the extracted table correctly represents the ground truth table, focusing on content accuracy, structural preservation, and information completeness. The extracted table was parsed from the rendered table. Disregard LaTeX-specific elements in the ground truth (e.g., comments, styling commands, font formatting) that have no effect on content or structure.

Ground Truth Table (LaTeX):
{gt_table}

Extracted Table:
{extracted_table}

Evaluate the extracted table using the following criteria:
1. Content accuracy: Are all cell values, headers, and data correctly preserved?
2. Structure preservation: Are all rows and columns present, and can each value be unambiguously mapped to its row/column headers? Broken or ambiguous associations count as errors.

Note: Different output formats (markdown, HTML, plain text) are acceptable as long as no information is lost. Apply this key test: Could a reader who sees ONLY the extracted table — without access to the ground truth — unambiguously reconstruct every cell-to-header mapping and all content of the original table? If not, consider the parsing as failed and assign a low score.

Assign a score from 0 to 10, where 10 is a perfect match.\
"""

NAIVE_PROMPT = """\
Rate how well the extracted table matches the ground truth table on a scale from 0 to 10.

Ground Truth Table:
{gt_table}

Extracted Table:
{extracted_table}\
"""


class TunedEvaluation(BaseModel):
    errors: list[str]
    score: int


class NaiveEvaluation(BaseModel):
    score: int


PROMPT_TEMPLATES = {
    "tuned": TUNED_PROMPT,
    "tuned_no_cot": TUNED_NO_COT_PROMPT,
    "naive": NAIVE_PROMPT,
}


def _response_format(variant: str) -> dict:
    cls = TunedEvaluation if variant == "tuned" else NaiveEvaluation
    return {
        "type": "json_schema",
        "json_schema": {
            "name": cls.__name__,
            "strict": True,
            "schema": cls.model_json_schema(),
        },
    }


def evaluate_table(
    client: OpenAI, model: str, variant: str, gt_table: str, extracted_table: str
) -> dict:
    prompt = PROMPT_TEMPLATES[variant].format(gt_table=gt_table, extracted_table=extracted_table)
    response_format = _response_format(variant)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                top_p=1,
                max_tokens=2048,
                response_format=response_format,
            )
            content = response.choices[0].message.content
            if variant == "tuned":
                ev = TunedEvaluation.model_validate_json(content)
                return {"score": ev.score, "errors": ev.errors}
            ev = NaiveEvaluation.model_validate_json(content)
            return {"score": ev.score, "errors": []}
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                raise


def main():
    if PROMPT_VARIANT not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown PROMPT_VARIANT: {PROMPT_VARIANT}")

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    total = sum(len(gt["extractions"]) for gt in data)
    print(f"Loaded {len(data)} GT tables, {total} extractions from {DATA_PATH}")

    todo = []
    for gt in data:
        for ext in gt["extractions"]:
            already = {(s["judge_model"], s["prompt_variant"]) for s in ext["llm_scores"]}
            for model in MODELS:
                if (model, PROMPT_VARIANT) not in already:
                    todo.append((gt, ext, model))

    print(f"{len(todo)} (extraction, model) pairs need evaluation with variant='{PROMPT_VARIANT}'")
    if not todo:
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required.")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=0,
        timeout=30,
    )

    lock = threading.Lock()
    done = 0

    def process(gt: dict, ext: dict, model: str) -> None:
        nonlocal done
        result = evaluate_table(
            client, model, PROMPT_VARIANT, gt["gt_table"], ext["extracted_table"]
        )
        score_entry = {
            "judge_model": model,
            "prompt_variant": PROMPT_VARIANT,
            "score": max(0, min(10, result["score"])),
            "errors": result["errors"],
        }
        ext_id = f"{ext['parser']}_{gt['gt_id']}"
        with lock:
            ext["llm_scores"].append(score_entry)
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            done += 1
            print(f"  [{done}/{len(todo)}] {model} {ext_id}: score={score_entry['score']}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process, gt, ext, model): (gt, ext, model)
            for gt, ext, model in todo
        }
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                gt, ext, model = futures[future]
                print(f"  FAILED {model} {ext['parser']}_{gt['gt_id']}: {exc}")

    print(f"Done. {done}/{len(todo)} pairs scored.")


if __name__ == "__main__":
    main()
