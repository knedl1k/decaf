#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GNU General Public License v3.0
# @knedl1k 2026

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import os
import math

INPUT_IMAGE_PATH = "10e_131_consume-spirit-b8117171-08a7-44f6-9c14-12e8b96c7632.png"
OUTPUT_FILENAME = "augmentation_grid.png"
IMG_SIZE = 224

def get_transforms(img_size):
    return A.Compose(
        [
            A.SafeRotate(limit=15.0, border_mode=cv2.BORDER_CONSTANT, p=0.7),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            

            A.CoarseDropout(num_holes_range=(1, 3), p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                        
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
            
            A.Normalize(),
            ToTensorV2(),
        ]
    )

def denormalize_image(img_tensor):
    # CHW -> HWC
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = std * img + mean
    img = np.clip(img, 0, 1) * 255
    return img.astype(np.uint8)

def create_grid(images, rows, cols):
    h, w, c = images[0].shape
    grid_img = np.zeros((rows * h, cols * w, c), dtype=np.uint8)

    for i, img in enumerate(images):
        if i >= rows * cols: break
        r = i // cols
        c = i % cols
        grid_img[r * h : (r + 1) * h, c * w : (c + 1) * w, :] = img
        
    return grid_img

def main():
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: Input image not found at {INPUT_IMAGE_PATH}")
        return NULL
    else:
        original_image = cv2.imread(INPUT_IMAGE_PATH)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    transform = get_transforms(IMG_SIZE)
    
    aug_images = []
    
    for i in range(4):
        augmented = transform(image=original_image)
        img_tensor = augmented["image"]
        
        img_vis = denormalize_image(img_tensor)
        
        img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        aug_images.append(img_bgr)

    final_grid = create_grid(aug_images, rows=2, cols=2)
    
    cv2.imwrite(OUTPUT_FILENAME, final_grid)

if __name__ == "__main__":
    main()
