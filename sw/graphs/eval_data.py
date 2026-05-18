#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="export data from evaluation (.npy)")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-od", "--output_dir", type=str, required=True)

    return parser.parse_args()


def export_history_to_csv(input: str, output: str) -> None:
    data = np.load(input, allow_pickle=True).item()

    hit_sims_name = np.array(data["hit_sims_name"])
    miss_sims_name = np.array(data["miss_sims_name"])
    hit_sims_exact = np.array(data["hit_sims_exact"])
    miss_sims_exact = np.array(data["miss_sims_exact"])

    bin_edges = np.linspace(0.0, 1.0, 41)

    hit_counts_name, _ = np.histogram(hit_sims_name, bins=bin_edges)
    miss_counts_name, _ = np.histogram(miss_sims_name, bins=bin_edges)

    hit_counts_exact, _ = np.histogram(hit_sims_exact, bins=bin_edges)
    miss_counts_exact, _ = np.histogram(miss_sims_exact, bins=bin_edges)

    hist_csv_path = os.path.join(output, "hist_data.csv")
    with open(hist_csv_path, "w", encoding="utf-8") as f:
        f.write("bin,hit_name,miss_name,hit_exact,miss_exact\n")
        for i in range(len(hit_counts_name)):
            f.write(
                f"{bin_edges[i]:.4f},{hit_counts_name[i]},{miss_counts_name[i]},{hit_counts_exact[i]},{miss_counts_exact[i]}\n"
            )
        f.write(f"{bin_edges[-1]:.4f},0,0,0,0\n")

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
    os.makedirs(args.output_dir, exist_ok=True)
    export_history_to_csv(args.input, args.output_dir)


if __name__ == "__main__":
    main()
