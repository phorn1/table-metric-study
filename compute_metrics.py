#!/usr/bin/env python3
"""Compute TEDS, TEDS-structure-only, GriTS, and SCORE metrics for all table pairs in all_tables.json."""

import json
from pathlib import Path

from scorers.normalize import normalize_table
from scorers.teds import TEDS
from scorers.grits import grits_from_html
from scorers.score_benchmark import score_from_html

DATA_PATH = Path(__file__).parent / "all_tables.json"


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    total = sum(len(gt["extractions"]) for gt in data)
    print(f"Loaded {len(data)} GT tables, {total} extractions from {DATA_PATH}")

    # Initialize TEDS scorers
    teds = TEDS(structure_only=False)
    teds_struct = TEDS(structure_only=True)

    i = 0
    for gt in data:
        gt_html = normalize_table(gt["gt_table_html"])

        for ext in gt["extractions"]:
            i += 1
            ext_id = f"{ext['parser']}_{gt['gt_id']}"

            extracted = ext["extracted_table"]
            pred_html = normalize_table(extracted)

            teds_val = teds.evaluate(pred_html, gt_html)
            teds_struct_val = teds_struct.evaluate(pred_html, gt_html)

            grits = grits_from_html(gt_html, pred_html)
            score = score_from_html(gt_html, pred_html)

            ext["metrics"] = {
                "teds": teds_val,
                "teds_structure": teds_struct_val,
                "grits_top": grits["grits_top"],
                "grits_con": grits["grits_con"],
                "score_content": score["score_content"],
                "score_content_shifted": score["score_content_shifted"],
                "score_index": score["score_index"],
            }

            m = ext["metrics"]
            print(f"  [{i}/{total}] {ext_id}: teds={m['teds']:.4f}, "
                  f"teds_struct={m['teds_structure']:.4f}, "
                  f"grits_top={m['grits_top']:.4f}, grits_con={m['grits_con']:.4f}, "
                  f"score_content={m['score_content']:.4f}, score_index={m['score_index']:.4f}")

    # Write back
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {len(data)} GT tables back to {DATA_PATH}")

    # Summary stats
    all_metrics = [ext["metrics"] for gt in data for ext in gt["extractions"]]

    def print_stat(label, key):
        vals = [m[key] for m in all_metrics]
        print(f"{label:24s} mean={sum(vals)/len(vals):.4f}")

    print_stat("TEDS:", "teds")
    print_stat("TEDS-structure:", "teds_structure")
    print_stat("GriTS-Top:", "grits_top")
    print_stat("GriTS-Con:", "grits_con")
    print_stat("SCORE content:", "score_content")
    print_stat("SCORE content shifted:", "score_content_shifted")
    print_stat("SCORE index:", "score_index")


if __name__ == "__main__":
    main()