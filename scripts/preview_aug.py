#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import copy
import albumentations as A
from pathlib import Path

from data import get_train_transforms


def generate_individual_aug_grid(image_path: str, output_dir: str, img_size: int = 512):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    pipeline = get_train_transforms(img_size)

    augs = {"00_original": A.Compose([])}

    idx = 1
    for transform in pipeline.transforms:
        transform_name = type(transform).__name__

        if transform_name in ["Normalize", "ToTensorV2"]:
            continue

        t_copy = copy.deepcopy(transform)

        t_copy.p = 1.0

        aug_name = f"{idx:02d}_{transform_name.lower()}"
        augs[aug_name] = A.Compose([t_copy])
        idx += 1

    for name, aug in augs.items():
        augmented_rgb = aug(image=img)["image"]
        augmented_bgr = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(save_path, augmented_bgr)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    for i in range(10):
        generate_individual_aug_grid(
            "/mnt/personal/adamej14/dataset/sum_177_shivan-dragon-b35b0fb6-2596-4da3-820e-df28172f209b.png",
            f"augs/0{i}/",
        )
