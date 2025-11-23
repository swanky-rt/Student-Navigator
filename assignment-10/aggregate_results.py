#!/usr/bin/env python3
"""
aggregate_results.py

Aggregate slowdown metrics + generate example transcripts PDF.

USAGE EXAMPLES
--------------
# If all CSVs are in the current directory:
    python aggregate_results.py

# If your CSVs are in an 'artifacts/' folder:
    python aggregate_results.py \
        --baseline artifacts/results_baseline.csv \
        --sudoku artifacts/results_sudoku.csv \
        --mdp artifacts/results_mdp.csv \
        --para-sudoku artifacts/results_defended_paraphrase_sudoku.csv \
        --para-mdp artifacts/results_defended_paraphrase_mdp.csv \
        --filter artifacts/results_filtering.csv \
        --table-csv aggregate_table.csv \
        --table-png aggregate_table.png \
        --pdf example_transcripts.pdf

This script will:
  • Build a per-id table with baseline vs Sudoku vs MDP tokens,
    overhead, and slowdown; save as CSV + PNG.
  • Compute mean slowdown for Sudoku and MDP and print them.
  • Generate a PDF with two example transcripts:
      - Example 1: Sudoku (id=7)
      - Example 2: MDP (id=6)
    showing baseline vs attacked outputs, plus paraphrase + filtering
    defense summaries (tokens, similarity, etc.).
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib.styles import getSampleStyleSheet


# ------------------------
# CONSTANTS: EXAMPLE IDS
# ------------------------
SUDOKU_ID = 7
MDP_ID = 6


def build_aggregate_table(
    baseline_path: str,
    sudoku_path: str,
    mdp_path: str,
    table_csv: str,
    table_png: str,
):
    print(f"[INFO] Loading baseline: {baseline_path}")
    df_base = pd.read_csv(baseline_path)

    print(f"[INFO] Loading Sudoku attack: {sudoku_path}")
    df_sudoku = pd.read_csv(sudoku_path)

    print(f"[INFO] Loading MDP attack: {mdp_path}")
    df_mdp = pd.read_csv(mdp_path)

    # Ensure IDs exist and are ints
    if "id" not in df_base.columns:
        df_base["id"] = range(len(df_base))
    if "id" not in df_sudoku.columns:
        df_sudoku["id"] = range(len(df_sudoku))
    if "id" not in df_mdp.columns:
        df_mdp["id"] = range(len(df_mdp))

    # Merge only on id and the metrics we care about
    df = df_base.merge(
        df_sudoku[["id", "reasoning_tokens", "cosine_similarity"]],
        on="id",
        suffixes=("_base", "_sudoku"),
    )
    df = df.merge(
        df_mdp[["id", "reasoning_tokens", "cosine_similarity"]],
        on="id",
    )

    # Rename mdp columns for clarity
    df = df.rename(
        columns={
            "reasoning_tokens": "reasoning_tokens_mdp",
            "cosine_similarity": "cosine_similarity_mdp",
        }
    )

    # Compute overheads + slowdown
    df["sudoku_overhead"] = df["reasoning_tokens_sudoku"] - df["reasoning_tokens_base"]
    df["mdp_overhead"] = df["reasoning_tokens_mdp"] - df["reasoning_tokens_base"]

    df["sudoku_slowdown"] = df["reasoning_tokens_sudoku"] / df["reasoning_tokens_base"]
    df["mdp_slowdown"] = df["reasoning_tokens_mdp"] / df["reasoning_tokens_base"]

    # Small, numeric-only table (by id)
    table_cols = [
        "id",
        "reasoning_tokens_base",
        "reasoning_tokens_sudoku",
        "reasoning_tokens_mdp",
        "sudoku_overhead",
        "mdp_overhead",
        "sudoku_slowdown",
        "mdp_slowdown",
    ]
    agg_table = df[table_cols].copy()

    # Save CSV
    agg_table.to_csv(table_csv, index=False)
    print(f"[OK] Saved aggregate table CSV: {table_csv}")

    # Compute and print mean slowdowns
    mean_sudoku = agg_table["sudoku_slowdown"].mean()
    mean_mdp = agg_table["mdp_slowdown"].mean()
    print(f"[METRIC] Mean Sudoku slowdown: {mean_sudoku:.3f}")
    print(f"[METRIC] Mean MDP slowdown   : {mean_mdp:.3f}")

    # Save table as PNG
    fig_height = 2 + 0.4 * len(agg_table)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=agg_table.round(3).values,
        colLabels=agg_table.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(agg_table.columns))))

    fig.tight_layout()
    fig.savefig(table_png, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved aggregate table PNG: {table_png}")

    return df, agg_table


def get_row_by_id(df: pd.DataFrame, idx: int):
    """Safe helper: get row with given id or None if missing."""
    match = df.loc[df["id"] == idx]
    if match.empty:
        return None
    return match.iloc[0]


def build_example_pdf(
    baseline_path: str,
    sudoku_path: str,
    mdp_path: str,
    para_sudoku_path: str,
    para_mdp_path: str,
    filter_path: str,
    pdf_path: str,
):
    print("[INFO] Building example transcripts PDF...")

    df_base = pd.read_csv(baseline_path)
    df_sudoku = pd.read_csv(sudoku_path)
    df_mdp = pd.read_csv(mdp_path)
    df_para_sudoku = pd.read_csv(para_sudoku_path)
    df_para_mdp = pd.read_csv(para_mdp_path)
    df_filter = pd.read_csv(filter_path)

    # Ensure id exists
    for d in (df_base, df_sudoku, df_mdp, df_para_sudoku, df_para_mdp, df_filter):
        if "id" not in d.columns:
            d["id"] = range(len(d))

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    h2_style = styles["Heading2"]
    body_style = styles["BodyText"]
    code_style = styles["Code"]

    story = []

    def add_example(example_title: str, attack_label: str, attack_df: pd.DataFrame,
                    para_df: pd.DataFrame, example_id: int):
        base_row = get_row_by_id(df_base, example_id)
        atk_row = get_row_by_id(attack_df, example_id)
        para_row = get_row_by_id(para_df, example_id)
        filt_row = get_row_by_id(df_filter, example_id)

        if base_row is None or atk_row is None:
            story.append(Paragraph(f"{example_title} (id={example_id}) — MISSING DATA", title_style))
            story.append(Spacer(1, 12))
            return

        # Heading
        story.append(Paragraph(f"{example_title} (id = {example_id})", title_style))
        story.append(Spacer(1, 12))

        q_text = str(base_row.get("question", ""))
        story.append(Paragraph("Question", h2_style))
        story.append(Preformatted(q_text, body_style))
        story.append(Spacer(1, 12))

        # Baseline output
        base_text = str(base_row.get("raw_generated_text", base_row.get("model_answer", "")))
        story.append(Paragraph("Baseline Output", h2_style))
        story.append(Preformatted(base_text, code_style))
        story.append(Spacer(1, 12))

        # Attacked output
        atk_text = str(atk_row.get("raw_generated_text", atk_row.get("model_answer", "")))
        story.append(Paragraph(f"{attack_label} Attacked Output", h2_style))
        story.append(Preformatted(atk_text, code_style))
        story.append(Spacer(1, 12))

        # Defense summaries
        story.append(Paragraph("Defense Effects (Summary)", h2_style))

        # Paraphrase defense summary
        if para_row is not None:
            p_tokens = para_row.get("reasoning_tokens", "")
            p_sim = para_row.get("cosine_similarity", "")
            p_time = para_row.get("elapsed_sec", "")
            p_snip = para_row.get("paraphrase_snippet", para_row.get("paraphrased_snippet", ""))

            p_text = (
                f"Paraphrase Defense:\n"
                f"  • reasoning_tokens = {p_tokens}\n"
                f"  • cosine_similarity = {p_sim}\n"
                f"  • elapsed_sec = {p_time}\n"
                f"  • paraphrased snippet: {p_snip}"
            )
            story.append(Preformatted(p_text, body_style))
        else:
            story.append(Preformatted("Paraphrase Defense: (no row found for this id)", body_style))

        story.append(Spacer(1, 12))

        # Filtering defense summary
        if filt_row is not None:
            f_tokens = filt_row.get("reasoning_tokens", "")
            f_sim = filt_row.get("cosine_similarity", "")
            f_time = filt_row.get("elapsed_sec", "")
            f_snip = filt_row.get("filtered_snippet", "")

            f_text = (
                f"Filtering Defense:\n"
                f"  • reasoning_tokens = {f_tokens}\n"
                f"  • cosine_similarity = {f_sim}\n"
                f"  • elapsed_sec = {f_time}\n"
                f"  • filtered snippet: {f_snip}"
            )
            story.append(Preformatted(f_text, body_style))
        else:
            story.append(Preformatted("Filtering Defense: (no row found for this id)", body_style))

        story.append(PageBreak())

    # Example 1: Sudoku
    add_example(
        example_title="Example 1 – Sudoku Overthinking Attack",
        attack_label="Sudoku",
        attack_df=df_sudoku,
        para_df=df_para_sudoku,
        example_id=SUDOKU_ID,
    )

    # Example 2: MDP
    add_example(
        example_title="Example 2 – MDP Overthinking Attack",
        attack_label="MDP",
        attack_df=df_mdp,
        para_df=df_para_mdp,
        example_id=MDP_ID,
    )

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    doc.build(story)
    print(f"[OK] Saved example transcripts PDF: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()

    # Input CSVs (no hardcoded 'artifacts/' in the names)
    parser.add_argument("--baseline", default="artifacts/results_baseline.csv", help="Baseline CSV")
    parser.add_argument("--sudoku", default="artifacts/results_sudoku.csv", help="Sudoku attack CSV")
    parser.add_argument("--mdp", default="artifacts/results_mdp.csv", help="MDP attack CSV")
    parser.add_argument(
        "--para-sudoku",
        default="artifacts/results_defended_paraphrase_sudoku.csv",
        help="Paraphrase defense CSV for Sudoku",
    )
    parser.add_argument(
        "--para-mdp",
        default="artifacts/results_defended_paraphrase_mdp.csv",
        help="Paraphrase defense CSV for MDP",
    )
    parser.add_argument(
        "--filter",
        default="artifacts/results_filtering.csv",
        help="Filtering defense CSV (used for both attacks)",
    )

    # Outputs
    parser.add_argument(
        "--table-csv", default="aggregate_table.csv", help="Output CSV for aggregate table"
    )
    parser.add_argument(
        "--table-png", default="aggregate_table.png", help="Output PNG for aggregate table"
    )
    parser.add_argument(
        "--pdf", default="example_transcripts.pdf", help="Output PDF for example transcripts"
    )

    args = parser.parse_args()

    # 1) Build aggregate table + PNG
    build_aggregate_table(
        baseline_path=args.baseline,
        sudoku_path=args.sudoku,
        mdp_path=args.mdp,
        table_csv=args.table_csv,
        table_png=args.table_png,
    )

    # 2) Build example transcript PDF
    build_example_pdf(
        baseline_path=args.baseline,
        sudoku_path=args.sudoku,
        mdp_path=args.mdp,
        para_sudoku_path=args.para_sudoku,
        para_mdp_path=args.para_mdp,
        filter_path=args.filter,
        pdf_path=args.pdf,
    )


if __name__ == "__main__":
    main()
