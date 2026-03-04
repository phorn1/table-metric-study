#!/usr/bin/env python3
"""Convert GT LaTeX tables to HTML using Claude Opus 4.6.

Reads all_tables.json, fills in gt_table_html for each GT entry by calling Claude.
Results are written back after each conversion so the script can be resumed.

Requires ANTHROPIC_API_KEY in .env or environment.
"""

import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = Path(__file__).parent / "all_tables.json"

SYSTEM_PROMPT = """\
You are a LaTeX-to-HTML table converter. Given a LaTeX tabular environment, \
output the equivalent HTML <table> element. Rules:
- Output ONLY the <table>...</table> HTML. No markdown fences, no explanation.
- Preserve all cell content as-is (including math notation — render it as plain text or keep simple symbols).
- Strip all formatting commands (\\textbf, \\textit, \\makecell, etc.) but keep their text content.
- For \\makecell{line1 \\\\ line2}, join with a space: "line1 line2".
- Strip \\hspace, \\vspace, and similar spacing commands.
"""


def convert_latex_to_html(client: anthropic.Anthropic, latex: str) -> str:
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Convert this LaTeX table to HTML:\n\n{latex}"}
        ],
    )
    return message.content[0].text.strip()


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    todo = [gt for gt in data if not gt.get("gt_table_html")]
    print(f"{len(todo)}/{len(data)} GT tables need HTML conversion")

    if not todo:
        return

    client = anthropic.Anthropic()

    for i, gt in enumerate(todo, 1):
        gt["gt_table_html"] = convert_latex_to_html(client, gt["gt_table"])

        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  [{i}/{len(todo)}] {gt['gt_id']}")


if __name__ == "__main__":
    main()