"""Rule-based partial table alignment, vendored from TabXEval (MIT License).

Source: https://github.com/CoRAL-ASU/TabXEval
        evaluation_pipeline/fuzzy_table_matching.py

Only the functions reachable from `merge_tables_fuzzy` (the entry point used
by the TabXEval evaluation pipeline) are vendored; the BERT-based mapping
variant (`merge_tables`) and its `bert_score` dependency are omitted.
`fuzzywuzzy` is replaced by its maintained drop-in fork `thefuzz`.
"""

from thefuzz import fuzz


def table_to_row_list(table_string):
    # Split the table into lines
    lines = table_string.strip().split('\n')

    # Find the first line that contains the table header (i.e., a line with '|')
    table_start_idx = next(
        (i for i, line in enumerate(lines) if '|' in line), None)

    # If no table is found, return an empty list
    if table_start_idx is None:
        return []

    # Process the header from the detected table start line
    header = lines[table_start_idx].strip().split('|')
    header = [col.strip() for col in header]

    # Initialize the output list with headers as the first row
    table_as_rows = [header]

    # Loop through each data row, skipping any separator rows and stopping at ``` or blank lines
    for line in lines[table_start_idx + 1:]:
        # Stop processing if the table ends
        if '```' in line or not line.strip():
            break

        # Skip lines that contain only '---'
        if '---' in line:
            continue

        # Split row values and handle empty cells as empty strings
        row_values = line.strip().split('|')
        row_values = [val.strip() if val.strip() else "" for val in row_values]

        # Ensure the row has the same number of columns as the header
        row = row_values + [""] * (len(header) - len(row_values))

        # Add the row to the table as rows
        table_as_rows.append(row)

    # modify the list of lists to remove all indexes with all empty strings
    table_as_rows = [row for row in table_as_rows if any(row)]
    table_struct = []
    idx_with_all_empty = set(list(range(len(table_as_rows[0]))))
    for row in table_as_rows:
        idx_with_empty = set([i for i, val in enumerate(row) if not val])
        idx_with_all_empty = idx_with_all_empty.intersection(idx_with_empty)

    for row in table_as_rows:
        table_struct.append([val for i, val in enumerate(
            row) if i not in idx_with_all_empty])
    return table_struct


def map_columns_fuzzy(table1, table2, column_mapping={}, row_mapping={}, threshold=100):
    fuzzy_scores = {}
    if not row_mapping:
        for idx1, col1 in enumerate(table1[0]):
            for idx2, col2 in enumerate(table2[0]):
                score = fuzz.ratio(col1, col2)
                fuzzy_scores[(idx1, idx2)] = score
        for (idx1, idx2), score in sorted(fuzzy_scores.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold and idx1 not in column_mapping and idx2 not in column_mapping.values():
                column_mapping[idx1] = idx2
        return column_mapping

    for idx1, col1 in enumerate(table1[0]):
        for idx2, col2 in enumerate(table2[0]):
            column_score = []
            for row1, row2 in row_mapping.items():
                if table1[row1][idx1] and table2[row2][idx2]:
                    score = fuzz.ratio(table1[row1][idx1], table2[row2][idx2])
                    column_score.append(score)
            if column_score:
                column_score = sum(column_score)/len(column_score)
            else:
                column_score = 0
            fuzzy_scores[(idx1, idx2)] = column_score
    for (idx1, idx2), score in sorted(fuzzy_scores.items(), key=lambda x: x[1], reverse=True):
        if score >= threshold and idx1 not in column_mapping and idx2 not in column_mapping.values():
            column_mapping[idx1] = idx2
    return column_mapping


def map_rows_fuzzy(table1, table2, column_mapping, threshold=100):
    row_mapping = {}
    row_fuzzy_scores = {}
    for idx1, row1 in enumerate(table1[1:]):
        for idx2, row2 in enumerate(table2[1:]):
            row_score = []
            for col1, col2 in column_mapping.items():
                if row1[col1] and row2[col2]:
                    score = fuzz.ratio(row1[col1], row2[col2])
                    row_score.append(score)

            row_fuzzy_scores[(idx1+1, idx2+1)] = sum(row_score) / \
                len(row_score) if row_score else 0

    for (idx1, idx2), score in sorted(row_fuzzy_scores.items(), key=lambda x: x[1], reverse=True):
        if score >= threshold and idx1 not in row_mapping and idx2 not in row_mapping.values():
            row_mapping[idx1] = idx2
    return row_mapping


def is_extra(label):
    """Check if the label is an 'extra' row or column."""
    return "extra" in str(label)


def get_cell_value(table, row, col):
    """Get the cell value, or 'none' if the row or column is 'extra'."""
    if is_extra(row) or is_extra(col):
        return "none"
    return table[row][col]


def merge_headers(table1, table2, column_mapping):
    """Merge the headers of two tables based on column mappings."""
    return [(get_cell_value(table1, 0, col1)+'.T1', get_cell_value(table2, 0, col2)+'.T2')
            for col1, col2 in column_mapping.items()]


def merge_rows(table1, table2, row1, row2, column_mapping):
    """Merge a pair of rows from two tables based on column mappings."""
    return [(get_cell_value(table1, row1, col1), get_cell_value(table2, row2, col2))
            for col1, col2 in column_mapping.items()]


def get_merged_tables(table1, table2, column_mapping, row_mapping):
    """Merge two tables based on column and row mappings."""
    # Merge headers
    merged_table = [merge_headers(table1, table2, column_mapping)]

    # Merge each row according to row mappings
    for row1, row2 in row_mapping.items():
        merged_table.append(merge_rows(
            table1, table2, row1, row2, column_mapping))

    return merged_table


def table_to_markdown(data, merged=False):
    markdown = ""
    # Loop through each row
    for idx, row in enumerate(data):
        # Convert each cell to the format: original_value (new_value)
        if idx == 1:
            mardown_row = " | ".join("---" for _ in row)
            markdown += f"| {mardown_row} |\n"
        markdown_row = " | ".join(
            [f"{cell[0]} / {cell[1]}" for cell in row]) if merged else " | ".join(row)
        markdown += f"| {markdown_row} |\n"
    return markdown


def merge_tables_fuzzy(table1, table2):
    try:
        table_1 = table_to_row_list(table1)
        table_2 = table_to_row_list(table2)
        if not table_1 or not table_2:
            return None, None
        column_mapping = {}
        row_mapping = {}
        column_mapping = map_columns_fuzzy(
            table_1, table_2, column_mapping, row_mapping)
        row_mapping = map_rows_fuzzy(table_1, table_2, column_mapping)
        column_mapping = map_columns_fuzzy(
            table_1, table_2, column_mapping, row_mapping)
        row_mapping = map_rows_fuzzy(table_1, table_2, column_mapping)
        merged_table = get_merged_tables(
            table_1, table_2, column_mapping, row_mapping)
        return table_to_markdown(merged_table, merged=True), merged_table
    except:
        return None, None
