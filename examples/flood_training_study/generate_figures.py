"""
Generate all 6 figures for the flood training study LaTeX paper.

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
import matplotlib.ticker as mticker
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


# ---------------------------------------------------------------------------
# Figure 1: Robustness vs flood level (3-panel, one per dataset)
# ---------------------------------------------------------------------------
def fig1_robustness_vs_flood(df):
    datasets = ["moons", "circles", "blobs"]
    fig, axes = plt.subplots(1, 3, figsize=(14.88, 3.88), sharey=True)

    for ax, ds in zip(axes, datasets):
        for do in [True, False]:
            sub = df[(df["dataset"] == ds) & (df["dropout"] == do)].sort_values("flood_level")
            ax.plot(
                sub["flood_level"],
                sub["mean_accuracy_drop"] * 100,
                marker="o",
                linestyle=DROPOUT_STYLES[do],
                color=DATASET_COLORS[ds],
                alpha=0.9 if do else 0.6,
                linewidth=2,
                markersize=6,
                label=DROPOUT_LABELS[do],
            )
        ax.set_title(ds.capitalize(), fontsize=14, fontweight="bold")
        ax.set_xlabel("Flood Level (b)", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("Mean Accuracy Drop (%)", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])

    fig.suptitle("SEU Robustness vs. Flood Level", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_robustness_vs_flood.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig1_robustness_vs_flood.png")


# ---------------------------------------------------------------------------
# Figure 2: Cost-benefit analysis (bar chart)
# ---------------------------------------------------------------------------
def fig2_cost_benefit(df):
    # Cross-dataset averages per flood level
    cross = (
        df.groupby("flood_level")
        .agg(
            mean_base_acc=("baseline_accuracy", "mean"),
            mean_drop=("mean_accuracy_drop", "mean"),
        )
        .reset_index()
    )

    baseline_drop = cross.loc[cross["flood_level"] == 0.0, "mean_drop"].values[0]
    baseline_acc = cross.loc[cross["flood_level"] == 0.0, "mean_base_acc"].values[0]

    cross["rel_improvement"] = (baseline_drop - cross["mean_drop"]) / baseline_drop * 100
    cross["acc_cost"] = (baseline_acc - cross["mean_base_acc"]) * 100

    flooded = cross[cross["flood_level"] > 0.0].copy()

    fig, ax1 = plt.subplots(figsize=(9.88, 5.88))

    x = np.arange(len(flooded))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        flooded["rel_improvement"],
        width,
        label="Robustness Improvement (%)",
        color=PALETTE[2],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        flooded["acc_cost"],
        width,
        label="Accuracy Cost (%)",
        color=PALETTE[3],
        edgecolor="black",
        linewidth=0.5,
    )

    ax1.set_xlabel("Flood Level", fontsize=12)
    ax1.set_ylabel("Relative Robustness Improvement (%)", fontsize=12, color=PALETTE[2])
    ax2.set_ylabel("Accuracy Cost (%)", fontsize=12, color=PALETTE[3])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"b={v:.2f}" for v in flooded["flood_level"]])

    # Highlight best (b=0.15)
    best_idx = flooded.index[flooded["flood_level"] == 0.15].tolist()
    if best_idx:
        pos = list(flooded.index).index(best_idx[0])
        ax1.annotate(
            "Best balance",
            xy=(pos - width / 2, flooded.iloc[pos]["rel_improvement"]),
            xytext=(pos - width / 2, flooded.iloc[pos]["rel_improvement"] + 1.5),
            ha="center",
            fontsize=9,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black"),
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.set_title("Cost-Benefit Analysis: Robustness Gain vs. Accuracy Cost", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_cost_benefit.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig2_cost_benefit.png")


# ---------------------------------------------------------------------------
# Figure 3: Training validation — final training loss vs. target flood level
# ---------------------------------------------------------------------------
def fig3_training_validation(df):
    fig, ax = plt.subplots(figsize=(9.88, 5.88))

    datasets = ["moons", "circles", "blobs"]
    for ds in datasets:
        for do in [True, False]:
            sub = df[(df["dataset"] == ds) & (df["dropout"] == do)].sort_values("flood_level")
            ax.plot(
                sub["flood_level"],
                sub["final_train_loss"],
                marker="s" if do else "^",
                linestyle=DROPOUT_STYLES[do],
                color=DATASET_COLORS[ds],
                alpha=0.85,
                linewidth=2,
                markersize=6,
                label=f"{ds} ({DROPOUT_LABELS[do]})",
            )

    # Reference line y = x (ideal: final loss = flood level when active)
    flood_levels = sorted(df["flood_level"].unique())
    ax.plot(flood_levels, flood_levels, "k--", alpha=0.4, linewidth=1, label="y = x (ideal)")

    ax.set_xlabel("Target Flood Level (b)", fontsize=12)
    ax.set_ylabel("Final Training Loss", fontsize=12)
    ax.set_title("Final Training Loss vs. Target Flood Level", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.set_xticks(flood_levels)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_training_validation.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig3_training_validation.png")


# ---------------------------------------------------------------------------
# Figure 4: Heatmap of accuracy drop across all 36 configurations
# ---------------------------------------------------------------------------
def fig4_heatmap(df):
    # Create label for each config
    df_heat = df.copy()
    df_heat["config"] = df_heat.apply(lambda r: f"{r['dataset']}\n{'dropout' if r['dropout'] else 'no drop'}", axis=1)
    df_heat["flood_str"] = df_heat["flood_level"].apply(lambda x: f"b={x:.2f}")

    pivot = df_heat.pivot_table(index="config", columns="flood_str", values="mean_accuracy_drop", aggfunc="first")
    # Order columns by flood level
    ordered_cols = [f"b={fl:.2f}" for fl in sorted(df["flood_level"].unique())]
    pivot = pivot[ordered_cols]

    # Order rows: datasets grouped together
    row_order = []
    for ds in ["blobs", "moons", "circles"]:
        for do_label in ["dropout", "no drop"]:
            row_order.append(f"{ds}\n{do_label}")
    pivot = pivot.reindex(row_order)

    fig, ax = plt.subplots(figsize=(11.28, 7.88))

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

    ax.set_title("Mean Accuracy Drop (%) Under SEU Injection", fontsize=14, fontweight="bold")
    ax.set_xlabel("Flood Level", fontsize=12)
    ax.set_ylabel("Configuration", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig4_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 5: Loss trajectories (2-panel)
#   Left: training loss at each flood level by dataset
#   Right: final converged loss vs. flood level
# ---------------------------------------------------------------------------
def fig5_loss_trajectories(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.93, 3.43))

    datasets = ["moons", "circles", "blobs"]

    # Left panel: final train loss vs flood level (with dropout only, for clarity)
    for ds in datasets:
        sub = df[(df["dataset"] == ds) & (df["dropout"] == True)].sort_values("flood_level")
        ax1.plot(
            sub["flood_level"],
            sub["final_train_loss"],
            marker="o",
            color=DATASET_COLORS[ds],
            linewidth=2,
            label=ds.capitalize(),
        )

    flood_levels = sorted(df["flood_level"].unique())
    ax1.plot(flood_levels, flood_levels, "k--", alpha=0.4, linewidth=1, label="y = x")
    ax1.set_xlabel("Flood Level (b)", fontsize=10)
    ax1.set_ylabel("Final Training Loss", fontsize=10)
    ax1.set_title("Training Loss Trajectories", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_xticks(flood_levels)

    # Right panel: final val loss vs flood level
    for ds in datasets:
        sub = df[(df["dataset"] == ds) & (df["dropout"] == True)].sort_values("flood_level")
        ax2.plot(
            sub["flood_level"],
            sub["final_val_loss"],
            marker="s",
            color=DATASET_COLORS[ds],
            linewidth=2,
            label=ds.capitalize(),
        )

    ax2.set_xlabel("Flood Level (b)", fontsize=10)
    ax2.set_ylabel("Final Validation Loss", fontsize=10)
    ax2.set_title("Converged Validation Loss", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_xticks(flood_levels)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_loss_trajectories.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig5_loss_trajectories.png")


# ---------------------------------------------------------------------------
# Figure 6: Comprehensive training dynamics (4-panel)
#   Top-left:  Baseline accuracy by dataset/dropout
#   Top-right: Final train loss vs final val loss
#   Bottom-left:  Relative robustness improvement vs flood level
#   Bottom-right: Per-bit vulnerability (from JSON data)
# ---------------------------------------------------------------------------
def fig6_training_dynamics(df, jdata):
    fig, axes = plt.subplots(2, 2, figsize=(9.93, 7.93))
    ax_acc, ax_loss, ax_rob, ax_bit = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    datasets = ["moons", "circles", "blobs"]

    # --- Top-left: Baseline accuracy by dataset/dropout ---
    x_pos = np.arange(len(datasets))
    width = 0.3
    for i, do in enumerate([True, False]):
        accs = []
        for ds in datasets:
            sub = df[(df["dataset"] == ds) & (df["dropout"] == do) & (df["flood_level"] == 0.0)]
            accs.append(sub["baseline_accuracy"].values[0] * 100)
        offset = -width / 2 + i * width
        ax_acc.bar(
            x_pos + offset,
            accs,
            width,
            label=DROPOUT_LABELS[do],
            color=PALETTE[4 + i],
            edgecolor="black",
            linewidth=0.5,
        )

    ax_acc.set_xticks(x_pos)
    ax_acc.set_xticklabels([d.capitalize() for d in datasets])
    ax_acc.set_ylabel("Baseline Accuracy (%)", fontsize=10)
    ax_acc.set_title("Baseline Accuracy", fontsize=12, fontweight="bold")
    ax_acc.legend(fontsize=8)
    ax_acc.set_ylim(70, 105)

    # --- Top-right: Train loss vs val loss scatter ---
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        ax_loss.scatter(
            sub["final_train_loss"],
            sub["final_val_loss"],
            color=DATASET_COLORS[ds],
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            label=ds.capitalize(),
        )
    lim_min = 0
    lim_max = max(df["final_train_loss"].max(), df["final_val_loss"].max()) * 1.1
    ax_loss.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, linewidth=1)
    ax_loss.set_xlabel("Final Train Loss", fontsize=10)
    ax_loss.set_ylabel("Final Val Loss", fontsize=10)
    ax_loss.set_title("Train vs. Validation Loss", fontsize=12, fontweight="bold")
    ax_loss.legend(fontsize=8)

    # --- Bottom-left: Relative robustness improvement ---
    cross = df.groupby("flood_level").agg(mean_drop=("mean_accuracy_drop", "mean")).reset_index()
    baseline_drop = cross.loc[cross["flood_level"] == 0.0, "mean_drop"].values[0]
    cross["rel_imp"] = (baseline_drop - cross["mean_drop"]) / baseline_drop * 100

    ax_rob.bar(
        range(len(cross)),
        cross["rel_imp"],
        color=[PALETTE[0] if fl == 0 else PALETTE[2] for fl in cross["flood_level"]],
        edgecolor="black",
        linewidth=0.5,
    )
    ax_rob.set_xticks(range(len(cross)))
    ax_rob.set_xticklabels([f"b={fl:.2f}" for fl in cross["flood_level"]], fontsize=8)
    ax_rob.set_ylabel("Relative Improvement (%)", fontsize=10)
    ax_rob.set_title("Cross-Dataset Robustness Improvement", fontsize=12, fontweight="bold")
    ax_rob.axhline(y=0, color="black", linewidth=0.5)

    # --- Bottom-right: Per-bit vulnerability ---
    bit_labels = {
        "0": "Bit 0\n(Sign)",
        "1": "Bit 1\n(Exp MSB)",
        "8": "Bit 8\n(Exp LSB)",
        "9": "Bit 9\n(Mant MSB)",
        "31": "Bit 31\n(Mant LSB)",
    }
    bit_ids = ["0", "1", "8", "9", "31"]

    # Average accuracy drop per bit across all configs
    bit_drops = {b: [] for b in bit_ids}
    for entry in jdata:
        if "seu_by_bit" in entry and entry["seu_by_bit"]:
            for b in bit_ids:
                if b in entry["seu_by_bit"]:
                    bit_drops[b].append(entry["seu_by_bit"][b]["accuracy_drop"] * 100)

    mean_bit_drops = [np.mean(bit_drops[b]) if bit_drops[b] else 0 for b in bit_ids]

    bars = ax_bit.bar(
        range(len(bit_ids)),
        mean_bit_drops,
        color=[PALETTE[1] if b == "1" else PALETTE[0] for b in bit_ids],
        edgecolor="black",
        linewidth=0.5,
    )
    ax_bit.set_xticks(range(len(bit_ids)))
    ax_bit.set_xticklabels([bit_labels[b] for b in bit_ids], fontsize=8)
    ax_bit.set_ylabel("Mean Accuracy Drop (%)", fontsize=10)
    ax_bit.set_title("Per-Bit Vulnerability", fontsize=12, fontweight="bold")

    fig.suptitle("Comprehensive Training Dynamics Analysis", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_training_dynamics.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig6_training_dynamics.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df, jdata = load_data()
    print(f"  CSV: {len(df)} rows, JSON: {len(jdata)} entries")

    print("Generating figures...")
    fig1_robustness_vs_flood(df)
    fig2_cost_benefit(df)
    fig3_training_validation(df)
    fig4_heatmap(df)
    fig5_loss_trajectories(df)
    fig6_training_dynamics(df, jdata)
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()
