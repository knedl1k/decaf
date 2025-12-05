#!/usr/bin/env python
# -*- coding: utf-8 -*-

import albumentations as A
import cv2
import os
from pathlib import Path
import logging

logging.basicConfig(format='LOG: %(message)s')
log = logging.getLogger(__name__)

INPUT_DIR = "images/"
OUTPUT_DIR = "data/"
IMG_SIZE = 224
AUGMENTATIONS_PER_CARD = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = A.Compose([
    A.SafeRotate(limit=15.0, p=0.7, border_mode=cv2.BORDER_CONSTANT), # crooked card 
    A.Perspective(scale=(0.05, 0.1), p=0.5), # view from an angle
        
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(5, 30), hole_width_range=(5, 30), p=0.3), # dust on the card
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5), # color change from reflections
    A.GaussianBlur(blur_limit=(1, 3), p=0.2), # blur

    # 224x224
    # maintains the ration
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(
        min_height=IMG_SIZE, 
        min_width=IMG_SIZE, 
        border_mode=cv2.BORDER_CONSTANT, 
        fill=[0, 0, 0] # black
    )
])

def main():
    image_paths = list(Path(INPUT_DIR).glob("*.png"))
    print(f"Found {len(image_paths)} images to generate from. Starting.")

    for file_path in image_paths:
        image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            log.warning(f"There has been an issue with {file_path.name}")
            continue

        if image.shape[2] == 4: #png check for alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        base_name = file_path.stem

        for i in range(AUGMENTATIONS_PER_CARD):
            try:
                augmented = transform(image=image_rgb)["image"]
                
                # transfer back to BGR, so OpenCV does not have any issues
                augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                output_filename = f"{base_name}_{i}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                cv2.imwrite(output_path, augmented_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
            except Exception as e:
                log.warning(f"The {base_name}: {e}")
    print(f"Dataset generated!")

if __name__ == '__main__':
    main()
