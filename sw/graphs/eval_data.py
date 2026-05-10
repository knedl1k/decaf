#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="export data from evaluation (.npy)")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-od", "--output_dir", type=str, required=True)

    return parser.parse_args()


def export_history_to_csv(input: str, output: str):
    data = np.load(input, allow_pickle=True).item()

    hit_sims = np.array(data["hit_sims_name"])
    miss_sims = np.array(data["miss_sims_name"])

    bin_edges = np.linspace(0.0, 1.0, 41)
    hit_counts, _ = np.histogram(hit_sims, bins=bin_edges)
    miss_counts, _ = np.histogram(miss_sims, bins=bin_edges)

    hist_csv_path = os.path.join(output, "hist_data.csv")
    with open(hist_csv_path, "w", encoding="utf-8") as f:
        f.write("bin,hit_count,miss_count\n")
        for i in range(len(hit_counts)):
            f.write(f"{bin_edges[i]:.4f},{hit_counts[i]},{miss_counts[i]}\n")

        f.write(f"{bin_edges[-1]:.4f},0,0\n")

    metrics = data["metrics"]
    total = metrics["total_images"]

    bar_csv_path = os.path.join(output, "bar_data.csv")
    with open(bar_csv_path, "w", encoding="utf-8") as f:
        f.write("k,name_match,exact_match\n")
        for k in [1, 3, 5]:
            name_acc = (metrics["name"][k] / total) * 100 if total > 0 else 0
            exact_acc = (metrics["exact"][k] / total) * 100 if total > 0 else 0
            f.write(f"Top-{k},{name_acc:.1f},{exact_acc:.1f}\n")

    print(f"Data exported to: {output}")


def main() -> None:
    args = parse_args()
    export_history_to_csv(args.input, args.output_dir)


if __name__ == "__main__":
    main()
