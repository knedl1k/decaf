#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
from pathlib import Path


def export_for_latex(npy_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npy_path, allow_pickle=True).item()

    # --- 1. Export Histogram Data (Frequencies) ---
    hit_sims = np.array(data["hit_sims_name"])
    miss_sims = np.array(data["miss_sims_name"])

    # Rozdělení na 40 binů v intervalu 0.0 až 1.0
    bin_edges = np.linspace(0.0, 1.0, 41)
    hit_counts, _ = np.histogram(hit_sims, bins=bin_edges)
    miss_counts, _ = np.histogram(miss_sims, bins=bin_edges)

    hist_csv_path = os.path.join(output_dir, "hist_data.csv")
    with open(hist_csv_path, "w", encoding="utf-8") as f:
        f.write("bin,hit_count,miss_count\n")
        # Exportujeme všechny biny (i nulové) pro zachování spojité osy X
        for i in range(len(hit_counts)):
            f.write(f"{bin_edges[i]:.4f},{hit_counts[i]},{miss_counts[i]}\n")

        # Přidání koncové hrany posledního binu, aby graf mohl korektně klesnout na 0
        f.write(f"{bin_edges[-1]:.4f},0,0\n")

    # --- 2. Export Bar Chart Data (Top-K Accuracies) ---
    metrics = data["metrics"]
    total = metrics["total_images"]

    bar_csv_path = os.path.join(output_dir, "bar_data.csv")
    with open(bar_csv_path, "w", encoding="utf-8") as f:
        f.write("k,name_match,exact_match\n")
        for k in [1, 3, 5]:
            name_acc = (metrics["name"][k] / total) * 100 if total > 0 else 0
            exact_acc = (metrics["exact"][k] / total) * 100 if total > 0 else 0
            f.write(f"Top-{k},{name_acc:.1f},{exact_acc:.1f}\n")

    print(f"Data successfully exported to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_file", type=str, default="./evaluation_data.npy")
    parser.add_argument("--out_dir", type=str, default="./latex_export")
    args = parser.parse_args()

    if Path(args.npy_file).exists():
        export_for_latex(args.npy_file, args.out_dir)
