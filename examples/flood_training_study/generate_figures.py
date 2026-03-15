"""
Generate figures for the flood training study LaTeX paper.

Reads data from data/comprehensive_results.csv and data/comprehensive_results.json,
produces publication-quality figures in paper_latex/figures/.

Usage:
    python generate_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIG_DIR = SCRIPT_DIR / "paper_latex" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "comprehensive_results.csv"
JSON_PATH = DATA_DIR / "comprehensive_results.json"

# --- Style ---
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = sns.color_palette("Set2")
DATASET_COLORS = {"moons": PALETTE[0], "circles": PALETTE[1], "blobs": PALETTE[2]}
DROPOUT_STYLES = {True: "-", False: "--"}
DROPOUT_LABELS = {True: "w/ dropout", False: "w/o dropout"}
DPI = 300


def load_data():
    df = pd.read_csv(CSV_PATH)
    with open(JSON_PATH, "r") as f:
        jdata = json.load(f)
    return df, jdata


def fig1_robustness_vs_flood(df):
    """3-panel line plot: mean accuracy drop vs flood level, one panel per dataset."""
    datasets = ["moons", "circles", "blobs"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, ds in zip(axes, datasets):
        for do in [True, False]:
            sub = df[(df["dataset"] == ds) & (df["dropout"] == do)].sort_values("flood_level")
            ax.plot(
                sub["flood_level"],
                sub["mean_accuracy_drop"] * 100,
                marker="o",
                linestyle=DROPOUT_STYLES[do],
                color=DATASET_COLORS[ds],
                alpha=0.9 if do else 0.55,
                linewidth=2,
                markersize=6,
                label=DROPOUT_LABELS[do],
            )
        ax.set_title(ds.capitalize(), fontsize=14, fontweight="bold")
        ax.set_xlabel("Flood Level (b)")
        if ax == axes[0]:
            ax.set_ylabel("Mean Accuracy Drop (%)")
        ax.legend(fontsize=9)
        ax.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])

    fig.suptitle("SEU Vulnerability vs. Flood Level", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig1_robustness_vs_flood.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


def fig2_training_loss(df):
    """Final training loss vs target flood level, one line per dataset (dropout=True only for clarity)."""
    datasets = ["moons", "circles", "blobs"]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for ds in datasets:
        sub = df[(df["dataset"] == ds) & (df["dropout"] == True)].sort_values("flood_level")
        ax.plot(
            sub["flood_level"],
            sub["final_train_loss"],
            marker="o",
            color=DATASET_COLORS[ds],
            linewidth=2,
            markersize=7,
            label=ds.capitalize(),
        )

    flood_levels = sorted(df["flood_level"].unique())
    ax.plot(flood_levels, flood_levels, "k--", alpha=0.4, linewidth=1, label="y = x (ideal)")

    ax.set_xlabel("Target Flood Level (b)")
    ax.set_ylabel("Final Training Loss")
    ax.set_title("Final Training Loss vs. Flood Level", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xticks(flood_levels)

    fig.tight_layout()
    out = FIG_DIR / "fig2_training_loss.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


def fig3_heatmap(df):
    """Heatmap of mean accuracy drop across all 36 configurations."""
    df_heat = df.copy()
    df_heat["config"] = df_heat.apply(lambda r: f"{r['dataset']}\n{'dropout' if r['dropout'] else 'no drop'}", axis=1)
    df_heat["flood_str"] = df_heat["flood_level"].apply(lambda x: f"b={x:.2f}")

    pivot = df_heat.pivot_table(index="config", columns="flood_str", values="mean_accuracy_drop", aggfunc="first")
    ordered_cols = [f"b={fl:.2f}" for fl in sorted(df["flood_level"].unique())]
    pivot = pivot[ordered_cols]

    row_order = []
    for ds in ["blobs", "moons", "circles"]:
        for do_label in ["dropout", "no drop"]:
            row_order.append(f"{ds}\n{do_label}")
    pivot = pivot.reindex(row_order)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        pivot * 100,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean Accuracy Drop (%)"},
        vmin=1.0,
        vmax=2.7,
    )
    ax.set_title("Mean Accuracy Drop (%) Under SEU Injection", fontsize=13, fontweight="bold")
    ax.set_xlabel("Flood Level")
    ax.set_ylabel("Configuration")

    fig.tight_layout()
    out = FIG_DIR / "fig3_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


def main():
    print("Loading data...")
    df, jdata = load_data()
    print(f"  CSV: {len(df)} rows, JSON: {len(jdata)} entries")

    print("Generating figures...")
    fig1_robustness_vs_flood(df)
    fig2_training_loss(df)
    fig3_heatmap(df)
    print("Done. 3 figures generated.")


if __name__ == "__main__":
    main()
