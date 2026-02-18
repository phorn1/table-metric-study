"""Convert all tables.json files from table_extraction/ into a single JSONL file."""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent / "table_extraction"
OUTPUT_FILE = Path(__file__).parent / "all_tables.jsonl"

SKIP_TABLE_IDS = {10, 11, 26, 29, 31, 34, 35}


def main():
    rows = []

    for tables_file in sorted(BASE_DIR.glob("*/*/tables.json")):
        parser_name = tables_file.parent.parent.name
        file_index = tables_file.parent.name

        with open(tables_file) as f:
            tables = json.load(f)

        for table in tables:
            table_index = table["index"]
            if table_index in SKIP_TABLE_IDS:
                continue
            row = {
                "id": f"{parser_name}_{file_index}_{table_index:02d}",
                "gt_table": table["gt_table"],
                "extracted_table": table["extracted_table"],
                "complexity": table["complexity"],
                "llm_scores": table.get("llm_scores", []),
            }
            rows.append(row)

    with open(OUTPUT_FILE, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()