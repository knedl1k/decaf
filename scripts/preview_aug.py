#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from pathlib import Path
import os


def generate_individual_aug_grid(image_path: str, output_dir: str, img_size: int = 512):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    augs = {
        "00_original": A.Compose([]),
        "01_affine": A.Compose(
            [
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-35, 35),
                    border_mode=cv2.BORDER_REPLICATE,
                    p=1.0,
                )
            ]
        ),
        "02_perspective": A.Compose([A.Perspective(scale=(0.05, 0.15), p=1.0)]),
        "03_sunflare": A.Compose(
            [
                A.RandomSunFlare(
                    src_radius=100,
                    num_flare_circles_range=(1, 2),
                    p=1.0,
                )
            ]
        ),
        "04_shadow": A.Compose([A.RandomShadow(num_shadows_limit=(1, 2), shadow_roi=(0, 0, 1, 1), p=1.0)]),
        "05_motionblur": A.Compose([A.MotionBlur(blur_limit=5, p=1.0)]),
        "06_glassblur": A.Compose([A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, p=1.0)]),
        "07_dropout": A.Compose(
            [
                A.CoarseDropout(
                    num_holes_range=(1, 3),
                    hole_height_range=(0.2, 0.45),
                    hole_width_range=(0.2, 0.45),
                    fill="random",
                    p=1.0,
                )
            ]
        ),
        "08_compression": A.Compose([A.ImageCompression(quality_range=(60, 100), p=1.0)]),
        "09_isonoise": A.Compose([A.ISONoise(p=1.0)]),
        "10_brightness": A.Compose([A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0)]),
        "11_hue": A.Compose([A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=30, val_shift_limit=20, p=1.0)]),
    }

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
