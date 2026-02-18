#!/usr/bin/env python3
"""Compute TEDS, TEDS-structure-only, and GriTS scores for all table pairs in all_tables.jsonl."""

import json
import sys

from scorers.normalize import normalized_html_table, normalized_markdown_table, normalized_latex_table
from scorers.teds import TEDS
from scorers.grits import grits_from_html
from scorers.score_benchmark import score_from_html, ZERO_SCORES


def main():
    input_path = "all_tables.jsonl"

    # Read all entries
    entries = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {input_path}")

    # Cache unique GT HTML normalizations
    gt_cache = {}
    for entry in entries:
        gt_html_raw = entry.get("gt_table_html", "")
        if gt_html_raw and gt_html_raw not in gt_cache:
            gt_cache[gt_html_raw] = normalized_html_table(gt_html_raw)
    print(f"Normalized {len(gt_cache)} unique GT HTML tables")

    # Initialize TEDS scorers
    teds = TEDS(structure_only=False)
    teds_struct = TEDS(structure_only=True)

    # Compute scores for each entry
    skipped = 0
    for i, entry in enumerate(entries, 1):
        gt_html_raw = entry.get("gt_table_html", "")
        if not gt_html_raw:
            entry["teds"] = 0.0
            entry["teds_structure_only"] = 0.0
            entry["grits_top"] = 0.0
            entry["grits_con"] = 0.0
            entry.update(ZERO_SCORES)
            skipped += 1
            print(f"  [{i}/{len(entries)}] {entry['id']}: SKIPPED (no gt_table_html)")
            continue

        gt_html = gt_cache[gt_html_raw]
        extracted = entry["extracted_table"]
        stripped = extracted.strip()
        if stripped.startswith("<table"):
            pred_html = normalized_html_table(extracted)
        elif "\\begin{tabular}" in stripped or "\\begin{table}" in stripped:
            pred_html = normalized_latex_table(extracted)
        else:
            pred_html = normalized_markdown_table(extracted)

        entry["teds"] = teds.evaluate(pred_html, gt_html)
        entry["teds_structure_only"] = teds_struct.evaluate(pred_html, gt_html)

        grits = grits_from_html(gt_html, pred_html)
        entry["grits_top"] = grits["grits_top"]
        entry["grits_con"] = grits["grits_con"]

        score = score_from_html(gt_html, pred_html)
        entry.update(score)

        print(f"  [{i}/{len(entries)}] {entry['id']}: teds={entry['teds']:.4f}, teds_struct={entry['teds_structure_only']:.4f}, grits_top={entry['grits_top']:.4f}, grits_con={entry['grits_con']:.4f}, score_content={entry['score_cell_content_acc']:.4f}, score_index={entry['score_cell_index_acc']:.4f}")

    # Write back
    with open(input_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if skipped:
        print(f"\nSkipped {skipped} entries without gt_table_html")
    print(f"Done. Wrote {len(entries)} entries back to {input_path}")

    # Summary stats
    def print_stat(label, key):
        vals = [e[key] for e in entries]
        print(f"{label:24s} mean={sum(vals)/len(vals):.4f}, min={min(vals):.4f}, max={max(vals):.4f}")

    print_stat("TEDS:", "teds")
    print_stat("TEDS-structure:", "teds_structure_only")
    print_stat("GriTS-Top:", "grits_top")
    print_stat("GriTS-Con:", "grits_con")
    print_stat("SCORE content_acc:", "score_cell_content_acc")
    print_stat("SCORE shifted_acc:", "score_shifted_content_acc")
    print_stat("SCORE index_acc:", "score_cell_index_acc")
    print_stat("SCORE TEDS:", "score_teds")
    print_stat("SCORE TEDS corrected:", "score_teds_corrected")


if __name__ == "__main__":
    main()
