#!/usr/bin/env python

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="generate graph from .npy")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)

    return parser.parse_args()


def plot_training_curves(history: Dict[str, list], save_dir: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scienceplots

    plt.style.use(["science", "ieee"])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 12,
            "font.size": 11,
            "legend.fontsize": 11,
            "axes.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
        }
    )

    c_loss = "#2c3e50"
    c_lr = "#7f8c8d"
    c_fmr = "#c0392b"
    c_pos = "#27ae60"
    c_neg = "#d35400"
    c_thresh = "#2980b9"
    c_acc = "#8e44ad"
    c_real = c_fmr
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Metrics", fontsize=16, fontweight="bold", y=0.98)
    epochs = history["epoch"]
    marker_style = "o" if len(epochs) <= 30 else ""
    msize = 4

    ax1 = axs[0, 0]
    ax1.plot(epochs, history["loss"], color=c_loss, marker=marker_style, markersize=msize, linewidth=2, label="Loss")
    ax1.set_title("Training Loss \\& Learning Rate")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xlim(left=1, right=60)
    ax1_lr = ax1.twinx()
    ax1_lr.plot(epochs, history["lr"], color=c_lr, linestyle="--", linewidth=1.5, label="Learning Rate")
    ax1_lr.set_ylabel("Learning Rate")
    ax1_lr.set_yscale("log")
    ax1_lr.spines["top"].set_visible(False)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_lr.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", frameon=True, edgecolor="white")

    ax2 = axs[0, 1]
    fmr = np.array(history["fmr"]) * 100
    ax2.plot(epochs, fmr, color=c_fmr, marker=marker_style, markersize=msize, linewidth=2)
    ax2.set_yscale("log")
    ax2.set_title("Validation FMR @ 95\\% TMR")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("FMR (\\%)")
    ax2.set_xlim(left=1, right=60)

    ax3 = axs[1, 0]
    pos = np.array(history["pos_sim"])
    pos_std = np.array(history.get("std_pos_sim", [0] * len(pos)))
    neg = np.array(history["neg_sim"])
    neg_std = np.array(history.get("std_neg_sim", [0] * len(neg)))
    thresh = np.array(history["threshold"])
    ax3.plot(epochs, pos, color=c_pos, marker=marker_style, markersize=msize, label="Avg Pos Sim", linewidth=2)
    ax3.fill_between(epochs, pos - pos_std, pos + pos_std, color=c_pos, alpha=0.15, linewidth=0)
    ax3.plot(epochs, neg, color=c_neg, marker=marker_style, markersize=msize, label="Avg Neg Sim", linewidth=2)
    ax3.fill_between(epochs, neg - neg_std, neg + neg_std, color=c_neg, alpha=0.15, linewidth=0)
    ax3.plot(epochs, thresh, color=c_thresh, linestyle="--", label="Threshold (95\\% TMR)", linewidth=2)
    ax3.set_title("Cosine Similarities \\& Threshold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Similarity (-1 to 1)")
    ax3.set_ylim(-0.2, 1.0)
    ax3.set_xlim(left=1, right=60)
    ax3.legend(loc="center right", frameon=True, edgecolor="white")

    ax4 = axs[1, 1]
    top1 = np.array(history["top1_acc"]) * 100
    real_top1 = np.array(history["real_top1"])
    ax4.plot(epochs, top1, color=c_acc, marker=marker_style, markersize=msize, linewidth=2)
    ax4.plot(epochs, real_top1, color=c_real, marker=marker_style, markersize=msize, linewidth=2)
    ax4.set_title("Top-1 Validation Accuracy")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy (\\%)")
    ax4.set_ylim(max(0, min(top1) - 5), 100.5)
    ax4.set_xlim(left=1, right=60)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path_pdf = os.path.join(save_dir + ".pdf")
    plot_path_png = os.path.join(save_dir + ".png")

    plt.savefig(plot_path_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(plot_path_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {save_dir}")


def save_csv(history: Dict[str, list], save_dir: str) -> None:
    np.save(save_dir, history)


def read_csv(save_dir: str):
    return np.load(save_dir, allow_pickle=True).item()


def export_history_to_csv(npy_path: str, output_csv: str):
    history = np.load(npy_path, allow_pickle=True).item()

    pos_sim = np.array(history["pos_sim"])
    pos_std = np.array(history.get("std_pos_sim", [0] * len(pos_sim)))

    neg_sim = np.array(history["neg_sim"])
    neg_std = np.array(history.get("std_neg_sim", [0] * len(neg_sim)))

    df = pd.DataFrame(
        {
            "epoch": history["epoch"],
            "loss": history["loss"],
            "lr": history["lr"],
            "fmr": np.array(history["fmr"]) * 100,
            "pos_sim": pos_sim,
            "pos_upper": pos_sim + pos_std,
            "pos_lower": pos_sim - pos_std,
            "neg_sim": neg_sim,
            "neg_upper": neg_sim + neg_std,
            "neg_lower": neg_sim - neg_std,
            "threshold": history["threshold"],
            "top1_acc": np.array(history["top1_acc"]) * 100,
            "real_top1": np.array(history["real_top1"]),
        }
    )

    df.to_csv(f"{output_csv}.csv", index=False)
    print(f"Data exported to {output_csv}.csv")


def main() -> None:
    args = parse_args()
    export_history_to_csv(args.input, args.output)
    # plot_training_curves(read_csv(args.input), args.output)


if __name__ == "__main__":
    main()
