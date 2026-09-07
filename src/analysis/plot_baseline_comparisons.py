#!/usr/bin/env python
"""Reproduce manuscript Figure 2 from shareable aggregate results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCAN_LABELS = {
    "CT": "CT", "MRI": "MRI", "PET": "PET", "ULTRASOUND": "Ultrasound",
    "XRAY": "X-ray", "MAMMO_TOMO": "Mammo", "NUCLEAR_MED": "Nuclear medicine",
}
TRANSITION_ORDER = [
    "MRI -> MRI", "MRI -> MAMMO_TOMO", "MAMMO_TOMO -> MRI",
    "MAMMO_TOMO -> MAMMO_TOMO", "ULTRASOUND -> ULTRASOUND", "Overall",
]
FIXED_POLICY = {
    "MRI -> MRI": "12-month guideline",
    "MRI -> MAMMO_TOMO": "6-month guideline",
    "MAMMO_TOMO -> MRI": "6-month guideline",
    "MAMMO_TOMO -> MAMMO_TOMO": "12-month guideline",
}
SERIES_STYLE = {
    "GRU-DSIMB": {"color": "#1b9e77", "marker": "o"},
    "Nearest guideline": {"color": "#7570b3", "marker": "D"},
    "Overall guideline": {"color": "#4d4d4d", "marker": "X"},
    "6-month guideline": {"color": "#d95f02", "marker": "s"},
    "12-month guideline": {"color": "#1f78b4", "marker": "^"},
    "24-month guideline": {"color": "#e31a1c", "marker": "P"},
    "Average": {"color": "#e7298a", "marker": "v"},
}
LABEL_OFFSET = {
    "GRU-DSIMB": -0.20, "Nearest guideline": 0.21, "Overall guideline": 0.21,
    "6-month guideline": -0.20, "12-month guideline": 0.21,
    "24-month guideline": -0.20,
}


def load_guideline_rows(path: Path) -> list[dict]:
    source = pd.read_csv(path)
    rows = []
    for transition in TRANSITION_ORDER:
        part = source.loc[source["transition"].eq(transition)]
        proposed = part.loc[part["model"].str.contains("ours", case=False, na=False)].iloc[0]
        guideline = part.loc[part["model"].eq("Guideline")].iloc[0]
        points = {
            "GRU-DSIMB": {"mae": float(proposed["MAE"]), "rmse": float(proposed["RMSE"])}
        }
        if transition in FIXED_POLICY:
            points[FIXED_POLICY[transition]] = {
                "mae": float(guideline["MAE"]), "rmse": float(guideline["RMSE"]),
            }
        elif transition == "ULTRASOUND -> ULTRASOUND":
            points["Nearest guideline"] = {
                "mae": float(guideline["MAE"]), "rmse": float(guideline["RMSE"]),
            }
            for months in (6, 12, 24):
                fixed = part.loc[part["model"].eq(f"{months}-month guideline")].iloc[0]
                points[f"{months}-month guideline"] = {
                    "mae": float(fixed["MAE"]), "rmse": float(fixed["RMSE"]),
                }
        else:
            points["Overall guideline"] = {
                "mae": float(guideline["MAE"]), "rmse": float(guideline["RMSE"]),
            }
        rows.append({"transition": transition, "n": int(proposed["n"]), "points": points})
    return rows


def load_average_rows(path: Path) -> list[dict]:
    source = pd.read_csv(path)
    rows = []
    for scan_type, part in source.groupby("scan_type", sort=False):
        proposed = part.loc[part["model"].str.contains("ours", case=False, na=False)].iloc[0]
        average = part.loc[part["model"].eq("Average baseline")].iloc[0]
        rows.append({
            "transition": str(scan_type),
            "n": int(proposed["n"]),
            "points": {
                "GRU-DSIMB": {"mae": float(proposed["MAE"]), "rmse": float(proposed["RMSE"])},
                "Average": {"mae": float(average["MAE"]), "rmse": float(average["RMSE"])},
            },
        })
    return rows


def row_axis_label(row: dict, is_guideline: bool) -> str:
    if row["transition"] == "Overall":
        return f"Overall\n(n={row['n']:,})"
    if is_guideline:
        previous_scan, next_scan = row["transition"].split(" -> ")
        base = f"{SCAN_LABELS.get(previous_scan, previous_scan)}\n→ {SCAN_LABELS.get(next_scan, next_scan)}"
    else:
        base = SCAN_LABELS.get(row["transition"], row["transition"])
    return f"{base}\n(n={row['n']:,})"


def draw_comparison_panel(ax, rows, metric, xlabel, is_guideline, show_ylabels=True):
    values = [point[metric] for row in rows for point in row["points"].values()]
    value_range = max(values) - min(values)
    ax.set_xlim(
        max(0, min(values) - max(0.18 * value_range, 0.8)),
        max(values) + max(0.15 * value_range, 0.65),
    )
    label_dx = max(0.018 * (ax.get_xlim()[1] - ax.get_xlim()[0]), 0.05)

    for y, row in enumerate(rows):
        row_values = [point[metric] for point in row["points"].values()]
        ax.plot([min(row_values), max(row_values)], [y, y], color="0.38", linewidth=2.2, zorder=1)
        for point_index, (name, point) in enumerate(row["points"].items()):
            style = SERIES_STYLE[name]
            value = point[metric]
            ax.scatter(value, y, s=42, marker=style["marker"], facecolor=style["color"],
                       edgecolor="black", linewidth=0.7, zorder=3)
            x_min, x_max = ax.get_xlim()
            x_range = x_max - x_min
            near_left_axis = value < x_min + 0.14 * x_range
            near_right_axis = value > x_min + 0.88 * x_range
            place_left = (name == "GRU-DSIMB" and not near_left_axis) or near_right_axis
            offset_y = -0.24 if point_index % 2 == 0 else 0.25
            if row["transition"] in {"ULTRASOUND -> ULTRASOUND", "MRI -> MRI", "Overall"}:
                offset_y = LABEL_OFFSET.get(name, offset_y)
            if row["transition"] == "MRI -> MAMMO_TOMO":
                if name == "GRU-DSIMB":
                    offset_y = -0.32
                elif name == "6-month guideline":
                    offset_y = 0.32
            text_x = value - label_dx if place_left else value + label_dx
            text_ha = "right" if place_left else "left"
            if is_guideline and name == "GRU-DSIMB":
                if row["transition"] == "MRI -> MRI":
                    text_x, text_ha = value + 1.2, "left"
                elif row["transition"] == "MRI -> MAMMO_TOMO":
                    text_x, text_ha = value - 0.9, "left"
                elif row["transition"] == "ULTRASOUND -> ULTRASOUND":
                    text_x, text_ha = value + 0.7, "right"
            ax.text(text_x, y + offset_y, f"{value:.4f}", color=style["color"],
                    fontsize=10, ha=text_ha, va="center")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_yticks(np.arange(len(rows)))
    if show_ylabels:
        ax.set_yticklabels([row_axis_label(row, is_guideline) for row in rows], fontsize=9.1)
        ax.get_yticklabels()[-1].set_fontweight("bold")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="x", labelsize=8.6)
    ax.spines[["top", "right"]].set_visible(False)


def legend_handles(names):
    return [Line2D([0], [0], linestyle="none", marker=SERIES_STYLE[name]["marker"],
                   markersize=6, markerfacecolor=SERIES_STYLE[name]["color"],
                   markeredgecolor="black", label=name) for name in names]


def make_vertical_figure(guideline_rows, average_rows):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0))
    draw_comparison_panel(axes[0, 0], guideline_rows, "mae", "MAE (months)", True, True)
    draw_comparison_panel(axes[0, 1], guideline_rows, "rmse", "RMSE (months)", True, False)
    draw_comparison_panel(axes[1, 0], average_rows, "mae", "MAE (months)", False, True)
    draw_comparison_panel(axes[1, 1], average_rows, "rmse", "RMSE (months)", False, False)
    names = ["GRU-DSIMB", "Nearest guideline", "Overall guideline", "6-month guideline",
             "12-month guideline", "24-month guideline", "Average"]
    fig.legend(handles=legend_handles(names), loc="upper center", ncol=4, frameon=False,
               fontsize=8.0, columnspacing=1.1, handletextpad=0.3)
    fig.tight_layout(pad=0.3, h_pad=1.0, w_pad=0.8)
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.06, top=0.89, hspace=0.28)
    return fig


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, default=repo_root / "aggregate_results")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "generated_figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = make_vertical_figure(
        load_guideline_rows(args.aggregate_dir / "guideline_comparison.csv"),
        load_average_rows(args.aggregate_dir / "average_comparison.csv"),
    )
    output = args.output_dir / "figure_2_baseline_comparisons.pdf"
    figure.savefig(output, format="pdf", bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
