"""TabXEval penalty scoring, ported from the TabXEval scoring notebook (MIT License).

Source: https://github.com/CoRAL-ASU/TabXEval
        eval_scores.ipynb (cells defining the penalty computation)

The notebook accumulates a penalty from extra/missing information (EI/MI) at
column, row, and cell level, plus data-type-specific penalties for partially
matching cells, and caps it at 1. The final TabXEval score reported by the
paper is `1 - penalty`, i.e. 1.0 means a perfect match.

All penalty weights are 1, as in the released notebook ("normal weights").
"""

import re

import pandas as pd

ALLOWED_DATA_TYPES = ["Numerical", "String",
                      "Bool", "Date", "List", "Time", "Others"]

PENALTIES = {"cell": 1, "row": 1, "col": 1, "MI": 1, "EI": 1, "Partial": 1}
DATATYPE_PENALTIES = {"numerical": 1, "bool": 1, "string": 1,
                      "date": 1, "time": 1, "list": 1, "others": 1}


def table_to_dict_list_comparison(table_string, suffix=""):
    """Notebook variant of the aligned-table parser: strips <think> blocks and
    does not require a `---` separator line after the header."""
    # If the string contains <think> and </think>, only process content after </think>
    if "</think>" in table_string:
        table_string = table_string.split("</think>", 1)[1]

    table_string = table_string.replace("markdown", "")
    # Split the table into lines
    lines = table_string.strip().split('\n')

    table_start_idx = next(
        (i for i, line in enumerate(lines) if '|' in line), None)

    # If no table is found, return an empty list
    if table_start_idx is None:
        return []

    # Process the header from the detected table start line
    header = lines[table_start_idx].strip().split('|')
    header = [col.strip() for col in header if col.strip()]

    table_as_dicts = []

    for line in lines[table_start_idx + 1:]:
        if '```' in line or not line.strip():
            break
        if '---' in line:
            continue

        row_values = line.strip().split('|')
        row_values = [val.strip() for val in row_values if val.strip()]

        row_dict = {}
        for i in range(len(header)):
            if i < len(row_values):
                row_dict[header[i]] = row_values[i]
            else:
                row_dict[header[i]] = None
        table_as_dicts.append(row_dict)

    return table_as_dicts


def _get_column_type_count(table, n_rows, type):
    if len(table) < n_rows:
        return 0
    mask = table.map(lambda x: (
        isinstance(x, tuple) and x[0] == type))
    type_count = mask.all(axis=0).sum()
    # Remove columns that match the condition
    table.drop(columns=table.columns[mask.all(axis=0)], inplace=True)
    cols_to_remove = []
    for col_header in table.columns:
        parts = col_header.split("/")
        if type == "MI" and ((len(parts) == 1 and "T1" in parts[0]) or (len(parts) == 2 and parts[1].strip() in ['-', ""])):
            cols_to_remove.append(col_header)
            type_count += 1
        elif type == "EI" and ((len(parts) == 1 and "T1" in parts[0]) or (len(parts) == 2 and parts[0].strip() in ['-', ""])):
            cols_to_remove.append(col_header)
            type_count += 1
    table.drop(columns=cols_to_remove, inplace=True)
    return type_count


def _get_row_type_count(table, n_cols, type):
    numm_cols = len(table.columns)
    if numm_cols < n_cols:
        return 0
    mask = table.map(lambda x: (
        isinstance(x, tuple) and x[0] == type))
    type_count = mask.all(axis=1).sum()
    # Remove rows that match the condition
    table.drop(index=table.index[mask.all(axis=1)], inplace=True)
    return type_count


def _get_cell_type_count(table, type):
    mask = table.map(lambda x: x == type or (
        isinstance(x, tuple) and x[0] == type))
    type_count = mask.sum().sum()
    # Remove cells that match the condition
    table[mask] = None
    return type_count


def parse_partial_cell_stats_table(data):
    lines = data.strip().split("\n")
    Type_index = None
    for i, line in enumerate(lines):
        if "Type" in line:
            Type_index = i
            break
    lines = lines[Type_index:]
    parsed_data = {}

    for line in lines[1:]:
        parts = [part.strip() for part in line.split("|")]
        parsed_data.update({
            parts[0]: [int(parts[1]), parts[2] if len(parts) > 2 else ""]
        })

    return parsed_data


def parse_nested_lists(input_str):
    list_contents = re.findall(r'\[(.*?)\]', input_str)
    result = []

    for content in list_contents:
        elements = []
        for elem in content.split(','):
            elem = elem.strip()
            try:
                elements.append(int(elem))
            except ValueError:
                elements.append(elem)
        result.append(elements)

    return result


def compute_penalty(record) -> float:
    """Compute the TabXEval penalty (0..1) for one processed table pair.

    `record` is the dict produced by the evaluation pipeline, holding
    `alignment`, `df_wo_em`, `ei_mi_table`, `partial_cell_delta_stats`,
    `partial_scores`, and `type_counts`.
    """
    if record['df_wo_em'].empty:
        return 0.0

    alignment = table_to_dict_list_comparison(record["alignment"])
    n_rows = len(alignment)
    n_cols = len(alignment[0])
    ei_mi_table = pd.DataFrame(record["ei_mi_table"])

    ei_column_count = _get_column_type_count(ei_mi_table, n_rows, "EI")
    mi_column_count = _get_column_type_count(ei_mi_table, n_rows, "MI")
    ei_row_count = _get_row_type_count(ei_mi_table, n_cols, "EI")
    mi_row_count = _get_row_type_count(ei_mi_table, n_cols, "MI")
    ei_cell_count = _get_cell_type_count(ei_mi_table, "EI")
    mi_cell_count = _get_cell_type_count(ei_mi_table, "MI")

    ei_score = (((ei_column_count)/n_cols)*PENALTIES['col']) + (((ei_row_count) /
                n_rows)*PENALTIES['row']) + (((ei_cell_count)/(n_rows*n_cols))*PENALTIES['cell'])
    mi_score = (((mi_column_count)/n_cols)*PENALTIES['col']) + (((mi_row_count) /
                n_rows)*PENALTIES['row']) + (((mi_cell_count)/(n_rows*n_cols))*PENALTIES['cell'])
    score = PENALTIES["EI"]*ei_score + PENALTIES["MI"]*mi_score

    partial_cell_statistics = parse_partial_cell_stats_table(
        record["partial_cell_delta_stats"])

    num_penalties = 0
    num_penalties += record['partial_scores']["Numerical"]
    if partial_cell_statistics['Numerical'][1] != "":
        numeric_cols = parse_nested_lists(
            partial_cell_statistics['Numerical'][1])
        for col in numeric_cols:
            if isinstance(col[-2], str):
                col[-2] = 0
            num_penalties += col[-2]

    string_penalties = partial_cell_statistics['String'][0]
    if partial_cell_statistics['String'][1] != "":
        # Zero penalty for abbreviations and semantically similar strings
        str_cols = parse_nested_lists(partial_cell_statistics['String'][1])
        for col in str_cols:
            string_penalties -= (col[-3] + col[-4])

    datetime_penalties = partial_cell_statistics['Date'][0] + \
        partial_cell_statistics['Time'][0]
    boolean_penalties = partial_cell_statistics['Boolean'][0]
    list_penalties = partial_cell_statistics['List'][0]
    other_penalties = record['type_counts']['Others']

    cell_penalties = num_penalties*DATATYPE_PENALTIES['numerical'] + string_penalties*DATATYPE_PENALTIES['string'] + datetime_penalties * \
        DATATYPE_PENALTIES['date'] + boolean_penalties * \
        DATATYPE_PENALTIES['bool'] + list_penalties*DATATYPE_PENALTIES['list'] + \
        other_penalties*DATATYPE_PENALTIES['others']
    cell_penalties = cell_penalties/(n_rows*n_cols)

    score += PENALTIES["Partial"]*cell_penalties*PENALTIES['cell']
    return min(1, score)
