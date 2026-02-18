#!/usr/bin/env python3
"""Convert GT LaTeX tables to HTML using Claude Opus 4.6.

Reads all_tables.jsonl, adds gt_table_html field (empty string initially),
then fills it in by calling Claude for each unique GT LaTeX table.
Results are written back incrementally so the script can be cancelled at any time.

Requires ANTHROPIC_API_KEY in .env or environment.
"""

import json
import os
import sys

import anthropic
from dotenv import load_dotenv

load_dotenv()

JSONL_PATH = "all_tables.jsonl"
MODEL = "claude-opus-4-6"
MAX_TOKENS = 8192

SYSTEM_PROMPT = """\
You are a LaTeX-to-HTML table converter. Given a LaTeX tabular environment, \
output the equivalent HTML <table> element. Rules:
- Output ONLY the <table>...</table> HTML. No markdown fences, no explanation.
- Preserve all cell content as-is (including math notation — render it as plain text or keep simple symbols).
- Strip all formatting commands (\\textbf, \\textit, \\makecell, etc.) but keep their text content.
- For \\makecell{line1 \\\\ line2}, join with a space: "line1 line2".
- Strip \\hspace, \\vspace, and similar spacing commands.
"""


def read_entries():
    entries = []
    with open(JSONL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_entries(entries):
    with open(JSONL_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def convert_latex_to_html(client, latex):
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Convert this LaTeX table to HTML:\n\n{latex}"}
        ],
    )
    return message.content[0].text.strip()


def main():
    entries = read_entries()
    print(f"Loaded {len(entries)} entries")

    # Step 1: Add gt_table_html="" to all entries that don't have it yet
    modified = False
    for entry in entries:
        if "gt_table_html" not in entry:
            entry["gt_table_html"] = ""
            modified = True
    if modified:
        write_entries(entries)
        print("Added gt_table_html field to all entries")

    # Step 2: Build unique GT mapping
    unique_gts = {}
    for entry in entries:
        latex = entry["gt_table"]
        if latex not in unique_gts:
            unique_gts[latex] = entry.get("gt_table_html", "")

    already_done = sum(1 for v in unique_gts.values() if v)
    todo = sum(1 for v in unique_gts.values() if not v)
    print(f"Unique GT tables: {len(unique_gts)} ({already_done} already converted, {todo} remaining)")

    if todo == 0:
        print("Nothing to do.")
        return

    client = anthropic.Anthropic()

    # Step 3: Convert each unique GT, write back after each one
    done = 0
    for latex, existing_html in list(unique_gts.items()):
        if existing_html:
            continue

        done += 1
        print(f"  [{done}/{todo}] Converting ({len(latex)} chars LaTeX)...", end=" ", flush=True)

        try:
            html_result = convert_latex_to_html(client, latex)
        except Exception as e:
            print(f"ERROR: {e}")
            html_result = ""

        unique_gts[latex] = html_result

        # Update all entries with this GT
        for entry in entries:
            if entry["gt_table"] == latex:
                entry["gt_table_html"] = html_result

        # Write back immediately
        write_entries(entries)
        preview = html_result[:80].replace('\n', ' ') if html_result else "(empty)"
        print(f"OK ({len(html_result)} chars): {preview}...")

    print(f"\nDone. Converted {done} unique GT tables.")


if __name__ == "__main__":
    main()