#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import random
import cv2
import numpy as np
from pathlib import Path

from data import MTGTrainDataset, get_train_transforms


def export_true_training_batch(ref_dir: str, output_dir: str, num_images: int = 16, img_size: int = 512):
    os.makedirs(output_dir, exist_ok=True)

    search_pattern = os.path.join(ref_dir, "*.[pjJ][npP][gG]*")
    all_files = [Path(p) for p in glob.glob(search_pattern)]

    label_map = {p.stem: 0 for p in all_files}

    transform = get_train_transforms(img_size)
    dataset = MTGTrainDataset(all_files, label_map, transform, img_size)

    indices = random.sample(range(len(dataset)), num_images)

    for i, idx in enumerate(indices):
        img_tensor, _ = dataset[idx]

        img_hwc = img_tensor.permute(1, 2, 0).numpy()

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_denorm = std * img_hwc + mean

        img_final = np.clip(img_denorm, 0, 1) * 255
        img_final = img_final.astype(np.uint8)

        img_out_bgr = cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(output_dir, f"batch_{i:02d}.jpg")
        cv2.imwrite(save_path, img_out_bgr)
        print(f"Saved: {save_path}")

    print(f"\nDone! Exported {num_images} to '{output_dir}'.")


if __name__ == "__main__":
    REFERENCE_DIR = "/mnt/personal/adamej14/dataset"
    OUTPUT_DIR = "full_augs/"

    export_true_training_batch(REFERENCE_DIR, OUTPUT_DIR)
