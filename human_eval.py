#!/usr/bin/env python3
"""Human evaluation UI for table extraction study.

Evaluators score each extraction 0-10 without seeing LLM scores,
but with access to the claude-opus error list.
"""

import base64
import json
import re
import subprocess
import tempfile
from collections import OrderedDict
from html import escape as html_escape
from pathlib import Path

import gradio as gr
import markdown as md_lib

LATEX_PREAMBLE = r"""\documentclass[varwidth=\maxdimen]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs,multirow,makecell,graphicx,array}
\usepackage{amsmath,amssymb,mathrsfs}
\usepackage{colortbl}
\usepackage[table]{xcolor}
\usepackage{adjustbox,caption,diagbox}
\usepackage{pifont}
"""


def render_latex_to_img(latex: str, temp_dir: Path) -> str:
    tex_path = temp_dir / "table.tex"
    pdf_path = temp_dir / "table.pdf"
    png_path = temp_dir / "table.png"

    doc = f"{LATEX_PREAMBLE}\\begin{{document}}\n\n{latex}\n\n\\end{{document}}\n"
    tex_path.write_text(doc, encoding="utf-8")

    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        capture_output=True, cwd=temp_dir, timeout=30,
    )
    if result.returncode != 0 or not pdf_path.exists():
        return "<p style='color:red'>LaTeX compilation failed</p>"

    subprocess.run(
        ["pdftoppm", "-png", "-r", "150", "-singlefile", str(pdf_path), str(png_path.with_suffix(""))],
        capture_output=True, timeout=10,
    )
    if not png_path.exists():
        return "<p style='color:red'>PNG conversion failed</p>"

    img_data = base64.b64encode(png_path.read_bytes()).decode()
    return (
        f'<div style="overflow-x:auto; max-width:100%">'
        f'<img src="data:image/png;base64,{img_data}" style="max-width:none">'
        f'</div>'
    )


def _is_latex_table(text: str) -> bool:
    return "\\begin{tabular}" in text or "\\begin{table}" in text


def render_extracted(text: str, raw: bool) -> str:
    if not text:
        return ""
    if raw:
        return (
            f'<pre style="overflow-x:auto; white-space:pre; font-family:monospace;'
            f' font-size:13px; padding:10px; background:#f8f8f8; border-radius:4px;'
            f' max-height:400px; overflow-y:auto">{html_escape(text)}</pre>'
        )
    stripped = text.strip()
    if _is_latex_table(stripped):
        with tempfile.TemporaryDirectory() as tmp:
            return render_latex_to_img(stripped, Path(tmp))
    is_html = stripped.lower().startswith("<table")
    table_html = text if is_html else md_lib.markdown(text, extensions=["tables"])
    return (
        f'<div style="overflow-x:auto">'
        f'<style>'
        f'.ext-tbl table {{ border-collapse:collapse; white-space:nowrap; }}'
        f'.ext-tbl td, .ext-tbl th {{ border:1px solid #ccc; padding:6px 10px; text-align:left; }}'
        f'.ext-tbl th {{ background:#f5f5f5; }}'
        f'</style>'
        f'<div class="ext-tbl">{table_html}</div>'
        f'</div>'
    )

JSONL_PATH = "all_tables.jsonl"


def load_grouped():
    """Load entries grouped by unique gt_table, preserving order."""
    entries = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    groups = OrderedDict()
    for e in entries:
        key = e["gt_table"]
        if key not in groups:
            groups[key] = []
        groups[key].append(e)
    return list(groups.values())


def save_and_reload(pending_scores, saved_eids):
    """Load fresh JSONL, apply pending scores, write back, return fresh groups.

    This avoids data loss from stale in-memory state by always reading
    the latest data from disk before writing.
    """
    entries = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    id_to_entry = {e["id"]: e for e in entries}
    for eid, score in pending_scores.items():
        if eid not in id_to_entry:
            continue
        entry = id_to_entry[eid]
        if "human_scores" not in entry:
            entry["human_scores"] = []
        if eid in saved_eids and entry["human_scores"]:
            entry["human_scores"][-1] = score
        else:
            entry["human_scores"].append(score)

    with open(JSONL_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    groups = OrderedDict()
    for e in entries:
        key = e["gt_table"]
        if key not in groups:
            groups[key] = []
        groups[key].append(e)
    return list(groups.values())


def get_opus_errors(entry):
    """Extract claude-opus error list from an entry."""
    return next(
        (
            s["errors"]
            for s in entry.get("llm_scores", [])
            if "claude-opus" in s.get("judge_model", "")
        ),
        [],
    )


def render_gt_html(group):
    """Render GT table to HTML image."""
    gt_latex = group[0]["gt_table"]
    with tempfile.TemporaryDirectory() as tmp:
        return render_latex_to_img(gt_latex, Path(tmp))


def render_extraction_card(entry, raw=False):
    """Render a single extraction card (table + errors, no scores)."""
    table_html = render_extracted(entry["extracted_table"], raw=raw)

    # TEDS scores
    teds = entry.get("teds")
    teds_s = entry.get("teds_structure_only")
    teds_parts = []
    if teds is not None:
        teds_parts.append(f"<b>TEDS:</b> {teds:.4f}")
    if teds_s is not None:
        teds_parts.append(f"<b>TEDS-struct:</b> {teds_s:.4f}")
    teds_html = ""
    if teds_parts:
        teds_html = (
            f'<div style="margin-top:10px; font-size:0.95em">'
            f'{" &nbsp;&nbsp; ".join(teds_parts)}'
            f"</div>"
        )

    # Claude-Opus errors
    errors = get_opus_errors(entry)
    errors_html = ""
    if errors:
        items = "".join(f"<li>{e}</li>" for e in errors)
        errors_html = (
            f'<div style="margin-top:10px; padding:8px; background:#fff8f0; '
            f'border-left:3px solid #e67e22; border-radius:4px">'
            f'<b style="color:#e67e22">Claude-Opus Errors:</b>'
            f"<ul style='margin:4px 0 0 16px; padding:0'>{items}</ul>"
            f"</div>"
        )
    return (
        f'<div style="border:1px solid #ddd; border-radius:6px; padding:12px; margin-bottom:8px">'
        f"{table_html}"
        f"{teds_html}"
        f"{errors_html}"
        f"</div>"
    )


def build_extraction_choices(group, pending_scores, saved_gt_indices, gt_idx):
    """Build radio choices: checkmark for session-scored, count of existing scores."""
    choices = []
    for entry in group:
        eid = entry["id"]
        n_scores = len(entry.get("human_scores", []))
        count_str = f" ({n_scores})" if n_scores > 0 else ""
        mark = " \u2713" if eid in pending_scores else ""
        choices.append(f"{eid}{count_str}{mark}")
    return choices


def get_current_score(entry, pending_scores):
    """Get score for an entry: from pending_scores (session) only, else default."""
    eid = entry["id"]
    if eid in pending_scores:
        return pending_scores[eid]
    return 5  # default – don't reveal previous scores


# ── UI ──────────────────────────────────────────────────────────────────────


def create_ui():
    groups = load_grouped()

    with gr.Blocks(title="Human Evaluation - Table Extraction") as app:
        gr.Markdown("# Human Evaluation - Table Extraction Study")

        # State
        gt_idx = gr.State(0)
        ext_idx = gr.State(0)
        pending_scores = gr.State({})  # {entry_id: int}
        saved_gt_indices = gr.State(set())  # set of entry IDs already saved in this session
        groups_state = gr.State(groups)

        # Header
        progress = gr.Markdown()

        # GT Table
        gr.Markdown("### Ground Truth Table")
        gt_html = gr.HTML()

        gr.Markdown("---")

        # Extraction card (directly below GT for easy comparison)
        ext_header = gr.Markdown()
        ext_card = gr.HTML()

        with gr.Row():
            score_slider = gr.Slider(
                minimum=0, maximum=10, step=1, value=5, label="Score (0 = unusable, 10 = perfect)"
            )
            raw_toggle = gr.Checkbox(label="Raw", value=False, scale=0, min_width=60)

        # Extraction selector + nav below the card
        ext_radio = gr.Radio(label="Select extraction", choices=[], interactive=True)

        with gr.Row():
            prev_ext_btn = gr.Button("< Prev Extraction")
            next_ext_btn = gr.Button("Next Extraction >")

        gr.Markdown("---")

        # GT nav
        with gr.Row():
            prev_gt_btn = gr.Button("< Prev GT", variant="secondary")
            save_next_btn = gr.Button("Save & Next GT >", variant="primary")

        # ── Helpers ──────────────────────────────────────────────────────

        def _render_page(g_idx, e_idx, p_scores, s_indices, grps, raw=False):
            """Return all UI outputs for a given GT index and extraction index."""
            group = grps[g_idx]
            n_ext = len(group)
            e_idx = max(0, min(e_idx, n_ext - 1))

            progress_text = f"**GT Table {g_idx + 1}/{len(grps)}** &nbsp;—&nbsp; Extraction {e_idx + 1}/{n_ext}"
            gt_rendered = render_gt_html(group)

            choices = build_extraction_choices(group, p_scores, s_indices, g_idx)
            current_choice = choices[e_idx] if choices else None

            entry = group[e_idx]
            ext_header_text = f"**{entry['id']}**"
            card_html = render_extraction_card(entry, raw=raw)
            score_val = get_current_score(entry, p_scores)

            return (
                g_idx,
                e_idx,
                progress_text,
                gt_rendered,
                gr.update(choices=choices, value=current_choice),
                ext_header_text,
                card_html,
                score_val,
            )

        # ── Callbacks ────────────────────────────────────────────────────

        def on_load(g_idx, e_idx, p_scores, s_indices, grps, raw):
            return _render_page(g_idx, e_idx, p_scores, s_indices, grps, raw)

        def on_raw_toggle(g_idx, e_idx, p_scores, s_indices, grps, raw):
            out = _render_page(g_idx, e_idx, p_scores, s_indices, grps, raw)
            # Only update ext_card (index 6)
            return out[6]

        def on_score_change(score, g_idx, e_idx, p_scores, grps):
            """Store score in pending_scores when slider changes."""
            group = grps[g_idx]
            entry = group[e_idx]
            p_scores[entry["id"]] = int(score)
            return p_scores

        def on_ext_radio(choice, g_idx, e_idx, p_scores, s_indices, grps, raw):
            """Switch to selected extraction via radio."""
            if choice is None:
                return g_idx, e_idx, "", "", "", 5, p_scores
            group = grps[g_idx]
            # Strip checkmark and score count to find ID
            clean = choice.replace(" \u2713", "")
            clean = re.sub(r' \(\d+\)$', '', clean)
            new_idx = next(
                (i for i, entry in enumerate(group) if entry["id"] == clean),
                e_idx,
            )
            out = _render_page(g_idx, new_idx, p_scores, s_indices, grps, raw)
            # out: g_idx, e_idx, progress, gt_html, radio_update, ext_header, card, score
            return (
                out[1],  # e_idx
                out[2],  # progress
                out[4],  # radio update
                out[5],  # ext_header
                out[6],  # card
                out[7],  # score
                p_scores,
            )

        def on_prev_ext(g_idx, e_idx, p_scores, s_indices, grps, raw):
            group = grps[g_idx]
            new_idx = (e_idx - 1) % len(group)
            return _render_page(g_idx, new_idx, p_scores, s_indices, grps, raw)

        def on_next_ext(g_idx, e_idx, p_scores, s_indices, grps, raw):
            group = grps[g_idx]
            new_idx = (e_idx + 1) % len(group)
            return _render_page(g_idx, new_idx, p_scores, s_indices, grps, raw)

        def on_save_and_next(g_idx, e_idx, p_scores, s_indices, grps, raw):
            """Save pending scores to JSONL, then advance to next GT."""
            grps = save_and_reload(p_scores, s_indices)
            s_indices.update(p_scores.keys())

            # Advance to next GT
            new_g_idx = (g_idx + 1) % len(grps)
            out = _render_page(new_g_idx, 0, p_scores, s_indices, grps, raw)
            return (
                out[0],  # g_idx
                out[1],  # e_idx
                p_scores,
                s_indices,
                grps,
                out[2],  # progress
                out[3],  # gt_html
                out[4],  # radio
                out[5],  # ext_header
                out[6],  # card
                out[7],  # score
            )

        def on_prev_gt(g_idx, e_idx, p_scores, s_indices, grps, raw):
            """Save and go to previous GT."""
            grps = save_and_reload(p_scores, s_indices)
            s_indices.update(p_scores.keys())

            new_g_idx = (g_idx - 1) % len(grps)
            out = _render_page(new_g_idx, 0, p_scores, s_indices, grps, raw)
            return (
                out[0],
                out[1],
                p_scores,
                s_indices,
                grps,
                out[2],
                out[3],
                out[4],
                out[5],
                out[6],
                out[7],
            )

        # ── Wiring ───────────────────────────────────────────────────────

        all_render_outputs = [
            gt_idx, ext_idx, progress, gt_html, ext_radio, ext_header, ext_card, score_slider
        ]

        full_gt_nav_outputs = [
            gt_idx, ext_idx, pending_scores, saved_gt_indices, groups_state,
            progress, gt_html, ext_radio, ext_header, ext_card, score_slider,
        ]

        state_inputs = [gt_idx, ext_idx, pending_scores, saved_gt_indices, groups_state, raw_toggle]

        # Score change -> update pending
        score_slider.release(
            on_score_change,
            inputs=[score_slider, gt_idx, ext_idx, pending_scores, groups_state],
            outputs=[pending_scores],
        )

        # Raw toggle -> re-render extraction card
        raw_toggle.change(
            on_raw_toggle,
            inputs=state_inputs,
            outputs=[ext_card],
        )

        # Extraction radio
        ext_radio.input(
            on_ext_radio,
            inputs=[ext_radio, gt_idx, ext_idx, pending_scores, saved_gt_indices, groups_state, raw_toggle],
            outputs=[ext_idx, progress, ext_radio, ext_header, ext_card, score_slider, pending_scores],
        )

        # Extraction prev/next
        prev_ext_btn.click(on_prev_ext, inputs=state_inputs, outputs=all_render_outputs)
        next_ext_btn.click(on_next_ext, inputs=state_inputs, outputs=all_render_outputs)

        # GT nav
        save_next_btn.click(on_save_and_next, inputs=state_inputs, outputs=full_gt_nav_outputs)
        prev_gt_btn.click(on_prev_gt, inputs=state_inputs, outputs=full_gt_nav_outputs)

        # Initial load
        app.load(on_load, inputs=state_inputs, outputs=all_render_outputs)

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch()
