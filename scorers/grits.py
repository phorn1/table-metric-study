"""
GriTS (Grid Table Similarity) metric.

Adapted from Microsoft's table-transformer implementation:
https://github.com/microsoft/table-transformer/blob/main/src/grits.py

Copyright (C) 2021 Microsoft Corporation

Modifications:
- Replaced fitz.Rect-based iou() with pure-Python implementation
- Replaced xml.etree.ElementTree with lxml.html for robust HTML parsing
- Removed grits_loc / bbox-related helpers (no spatial data available)
"""
import itertools
from difflib import SequenceMatcher
from collections import defaultdict

import numpy as np
from lxml import html


def compute_fscore(num_true_positives, num_true, num_positives):
    if num_positives > 0:
        precision = num_true_positives / num_positives
    else:
        precision = 1
    if num_true > 0:
        recall = num_true_positives / num_true
    else:
        recall = 1

    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0

    return fscore, precision, recall


def initialize_DP(sequence1_length, sequence2_length):
    scores = np.zeros((sequence1_length + 1, sequence2_length + 1))
    pointers = np.zeros((sequence1_length + 1, sequence2_length + 1))

    for seq1_idx in range(1, sequence1_length + 1):
        pointers[seq1_idx, 0] = -1

    for seq2_idx in range(1, sequence2_length + 1):
        pointers[0, seq2_idx] = 1

    return scores, pointers


def traceback(pointers):
    seq1_idx = pointers.shape[0] - 1
    seq2_idx = pointers.shape[1] - 1
    aligned_sequence1_indices = []
    aligned_sequence2_indices = []
    while not (seq1_idx == 0 and seq2_idx == 0):
        if pointers[seq1_idx, seq2_idx] == -1:
            seq1_idx -= 1
        elif pointers[seq1_idx, seq2_idx] == 1:
            seq2_idx -= 1
        else:
            seq1_idx -= 1
            seq2_idx -= 1
            aligned_sequence1_indices.append(seq1_idx)
            aligned_sequence2_indices.append(seq2_idx)

    aligned_sequence1_indices = aligned_sequence1_indices[::-1]
    aligned_sequence2_indices = aligned_sequence2_indices[::-1]

    return aligned_sequence1_indices, aligned_sequence2_indices


def align_1d(sequence1, sequence2, reward_lookup, return_alignment=False):
    sequence1_length = len(sequence1)
    sequence2_length = len(sequence2)

    scores, pointers = initialize_DP(sequence1_length, sequence2_length)

    for seq1_idx in range(1, sequence1_length + 1):
        for seq2_idx in range(1, sequence2_length + 1):
            reward = reward_lookup[sequence1[seq1_idx - 1] + sequence2[seq2_idx - 1]]
            diag_score = scores[seq1_idx - 1, seq2_idx - 1] + reward
            skip_seq2_score = scores[seq1_idx, seq2_idx - 1]
            skip_seq1_score = scores[seq1_idx - 1, seq2_idx]

            max_score = max(diag_score, skip_seq1_score, skip_seq2_score)
            scores[seq1_idx, seq2_idx] = max_score
            if diag_score == max_score:
                pointers[seq1_idx, seq2_idx] = 0
            elif skip_seq1_score == max_score:
                pointers[seq1_idx, seq2_idx] = -1
            else:
                pointers[seq1_idx, seq2_idx] = 1

    score = scores[-1, -1]

    if not return_alignment:
        return score

    sequence1_indices, sequence2_indices = traceback(pointers)

    return sequence1_indices, sequence2_indices, score


def align_2d_outer(true_shape, pred_shape, reward_lookup):
    scores, pointers = initialize_DP(true_shape[0], pred_shape[0])

    for row_idx in range(1, true_shape[0] + 1):
        for col_idx in range(1, pred_shape[0] + 1):
            reward = align_1d(
                [(row_idx - 1, tcol) for tcol in range(true_shape[1])],
                [(col_idx - 1, prow) for prow in range(pred_shape[1])],
                reward_lookup,
            )
            diag_score = scores[row_idx - 1, col_idx - 1] + reward
            same_row_score = scores[row_idx, col_idx - 1]
            same_col_score = scores[row_idx - 1, col_idx]

            max_score = max(diag_score, same_col_score, same_row_score)
            scores[row_idx, col_idx] = max_score
            if diag_score == max_score:
                pointers[row_idx, col_idx] = 0
            elif same_col_score == max_score:
                pointers[row_idx, col_idx] = -1
            else:
                pointers[row_idx, col_idx] = 1

    score = scores[-1, -1]

    aligned_true_indices, aligned_pred_indices = traceback(pointers)

    return aligned_true_indices, aligned_pred_indices, score


def factored_2dmss(true_cell_grid, pred_cell_grid, reward_function):
    pre_computed_rewards = {}
    transpose_rewards = {}
    for trow, tcol, prow, pcol in itertools.product(
        range(true_cell_grid.shape[0]),
        range(true_cell_grid.shape[1]),
        range(pred_cell_grid.shape[0]),
        range(pred_cell_grid.shape[1]),
    ):
        reward = reward_function(true_cell_grid[trow, tcol], pred_cell_grid[prow, pcol])

        pre_computed_rewards[(trow, tcol, prow, pcol)] = reward
        transpose_rewards[(tcol, trow, pcol, prow)] = reward

    num_pos = pred_cell_grid.shape[0] * pred_cell_grid.shape[1]
    num_true = true_cell_grid.shape[0] * true_cell_grid.shape[1]

    true_row_nums, pred_row_nums, row_pos_match_score = align_2d_outer(
        true_cell_grid.shape[:2], pred_cell_grid.shape[:2], pre_computed_rewards
    )

    true_column_nums, pred_column_nums, col_pos_match_score = align_2d_outer(
        true_cell_grid.shape[:2][::-1],
        pred_cell_grid.shape[:2][::-1],
        transpose_rewards,
    )

    pos_match_score_upper_bound = min(row_pos_match_score, col_pos_match_score)
    upper_bound_score, _, _ = compute_fscore(pos_match_score_upper_bound, num_pos, num_true)

    positive_match_score = 0
    for true_row_num, pred_row_num in zip(true_row_nums, pred_row_nums):
        for true_column_num, pred_column_num in zip(true_column_nums, pred_column_nums):
            positive_match_score += pre_computed_rewards[
                (true_row_num, true_column_num, pred_row_num, pred_column_num)
            ]

    fscore, precision, recall = compute_fscore(positive_match_score, num_true, num_pos)

    return fscore, precision, recall, upper_bound_score


def lcs_similarity(string1, string2):
    if len(string1) == 0 and len(string2) == 0:
        return 1
    s = SequenceMatcher(None, string1, string2)
    lcs = "".join(
        [string1[block.a : (block.a + block.size)] for block in s.get_matching_blocks()]
    )
    return 2 * len(lcs) / (len(string1) + len(string2))


def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def cells_to_grid(cells, key='cell_text'):
    if len(cells) == 0:
        return [[]]
    num_rows = max([max(cell['row_nums']) for cell in cells]) + 1
    num_columns = max([max(cell['column_nums']) for cell in cells]) + 1
    cell_grid = [["" for _ in range(num_columns)] for _ in range(num_rows)]
    for cell in cells:
        for row_num in cell['row_nums']:
            for column_num in cell['column_nums']:
                cell_grid[row_num][column_num] = cell[key]

    return cell_grid


def cells_to_relspan_grid(cells):
    if len(cells) == 0:
        return [[]]
    num_rows = max([max(cell['row_nums']) for cell in cells]) + 1
    num_columns = max([max(cell['column_nums']) for cell in cells]) + 1
    cell_grid = [[[0, 0, 1, 1] for _ in range(num_columns)] for _ in range(num_rows)]
    for cell in cells:
        min_row_num = min(cell['row_nums'])
        min_column_num = min(cell['column_nums'])
        max_row_num = max(cell['row_nums']) + 1
        max_column_num = max(cell['column_nums']) + 1
        for row_num in cell['row_nums']:
            for column_num in cell['column_nums']:
                cell_grid[row_num][column_num] = [
                    min_column_num - column_num,
                    min_row_num - row_num,
                    max_column_num - column_num,
                    max_row_num - row_num,
                ]

    return cell_grid


def grits_top(true_relative_span_grid, pred_relative_span_grid):
    return factored_2dmss(
        true_relative_span_grid, pred_relative_span_grid, reward_function=iou
    )


def grits_con(true_text_grid, pred_text_grid):
    return factored_2dmss(
        true_text_grid, pred_text_grid, reward_function=lcs_similarity
    )


def html_to_cells(table_html):
    """Parse an HTML table string into a list of cell dicts using lxml."""
    if not table_html:
        return None
    try:
        doc = html.fromstring(table_html)
    except Exception:
        return None

    tables = doc.xpath('//table')
    if not tables:
        return None

    table = tables[0]
    table_cells = []
    occupied_columns_by_row = defaultdict(set)
    current_row = -1

    for elem in table.iter():
        if elem.tag == 'tr':
            current_row += 1

        if elem.tag in ('td', 'th'):
            colspan = int(elem.get('colspan', '1'))
            rowspan = int(elem.get('rowspan', '1'))
            row_nums = list(range(current_row, current_row + rowspan))
            try:
                max_occupied = max(occupied_columns_by_row[current_row])
                current_column = min(
                    set(range(max_occupied + 2)).difference(
                        occupied_columns_by_row[current_row]
                    )
                )
            except ValueError:
                current_column = 0
            column_nums = list(range(current_column, current_column + colspan))
            for rn in row_nums:
                occupied_columns_by_row[rn].update(column_nums)

            cell_text = elem.text_content().strip()
            table_cells.append({
                'row_nums': row_nums,
                'column_nums': column_nums,
                'cell_text': cell_text,
            })

    return table_cells if table_cells else None


def grits_from_html(true_html, pred_html):
    """Compute GriTS_Top and GriTS_Con for two HTML table strings."""
    true_cells = html_to_cells(true_html)
    pred_cells = html_to_cells(pred_html)

    if true_cells is None or pred_cells is None:
        return {'grits_top': 0.0, 'grits_con': 0.0}

    true_topology_grid = np.array(cells_to_relspan_grid(true_cells))
    pred_topology_grid = np.array(cells_to_relspan_grid(pred_cells))
    true_text_grid = np.array(cells_to_grid(true_cells, key='cell_text'), dtype=object)
    pred_text_grid = np.array(cells_to_grid(pred_cells, key='cell_text'), dtype=object)

    metrics = {}

    metrics['grits_top'], _, _, _ = grits_top(true_topology_grid, pred_topology_grid)
    metrics['grits_con'], _, _, _ = grits_con(true_text_grid, pred_text_grid)

    return metrics
