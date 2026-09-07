#!/usr/bin/env python
"""Reproduce manuscript Figure 3 from shareable aggregate results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURE_ORDER = ["Imaging", "Mastectomy", "Biomarker"]
MODEL_ORDER = ["Linear", "XGBoost", "MLP", "GRU", "LSTM", "Decoder-only Transformer"]


def plot_feature_group_contribution(source: pd.DataFrame):
    model_labels = [label for label in MODEL_ORDER if (source["model_display"] == label).any()]
    palette = dict(zip(model_labels, sns.color_palette("Set2", n_colors=len(model_labels))))
    metric_specs = [
        ("micro_auc_contribution_mean", "Micro-AUC"),
        ("mae_reduction_mean", "MAE"),
    ]
    contribution_long = source.melt(
        id_vars=["model", "model_display", "feature_group", "n_pairs"],
        value_vars=[metric for metric, _ in metric_specs],
        var_name="metric",
        value_name="contribution",
    )
    contribution_long["metric_label"] = contribution_long["metric"].map(dict(metric_specs))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=False, constrained_layout=True)

    for ax, (metric_name, _) in zip(axes, metric_specs):
        part = contribution_long.loc[contribution_long["metric"].eq(metric_name)]
        sns.barplot(
            data=part, y="feature_group", x="contribution", hue="model_display",
            order=FEATURE_ORDER, hue_order=model_labels, ax=ax, palette=palette, alpha=0.82,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(part["metric_label"].iloc[0], loc="left", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", which="both", left=False)
        for spine in ("bottom", "top", "right", "left"):
            ax.spines[spine].set_visible(False)

        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        for container in ax.containers:
            for bar in container:
                width = bar.get_width()
                if pd.isna(width):
                    continue
                ax.text(max(width, 0) + 0.015 * x_range, bar.get_y() + bar.get_height() / 2,
                        f"{width:.4f}", va="center", ha="left", fontsize=8)
        if ax.legend_ is not None:
            ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    return fig


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=repo_root / "aggregate_results" / "feature_group_contributions.csv")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "generated_figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = plot_feature_group_contribution(pd.read_csv(args.input))
    output = args.output_dir / "figure_3_feature_group_contributions.pdf"
    figure.savefig(output, format="pdf", bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
