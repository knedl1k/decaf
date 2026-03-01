#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import math
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Tuple, Callable, Union

# prevent OpenCV from spawning too many threads in DataLoader workers
cv2.setNumThreads(0)


def get_inference_transforms(img_size: int) -> A.Compose:
    """Transforms for validation gallery, database creation, and single-image inference."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def get_train_transforms(img_size: int) -> A.Compose:
    """Augmentation pipeline for training and query validation."""
    return A.Compose(
        [
            # geometric deformations
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=35, border_mode=cv2.BORDER_REPLICATE, p=0.8
            ),
            A.Perspective(scale=(0.05, 0.15), p=0.5),
            # lighting & sleeve glare simulation
            A.RandomSunFlare(src_radius=100, num_flare_circles_range=(1, 2), p=0.3),
            A.RandomShadow(num_shadows_limit=(1, 2), shadow_roi=(0, 0, 1, 1), p=0.2),
            # mera artifacts
            A.CoarseDropout(
                hole_height_range=(int(img_size * 0.1), int(img_size * 0.25)),
                hole_width_range=(int(img_size * 0.1), int(img_size * 0.25)),
                p=0.4,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.ImageCompression(quality_range=(60, 100), p=0.2),
            A.ISONoise(p=0.2),
            # colors
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
            A.CLAHE(p=0.3),
            A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Unified image loading to handle 16-bit, RGBA, and standard BGR images."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Image not found or unreadable: {path}")

    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)

    return image


def apply_random_background(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resizes the card and places it on a random noisy background, handling alpha channels."""
    h, w = image.shape[:2]
    scale = (target_size * 0.9) / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_card = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # generate noisy background
    base_color = np.random.randint(20, 230, (1, 1, 3), dtype=np.uint8)
    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * base_color
    noise = np.random.normal(0, 4, (target_size, target_size, 3))
    canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)

    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2

    # blend using alpha channel if present
    if len(resized_card.shape) == 3 and resized_card.shape[2] == 4:
        bgr = resized_card[:, :, :3]
        alpha = resized_card[:, :, 3] / 255.0
        alpha_3d = np.expand_dims(alpha, axis=2)
        roi = canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w]
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (bgr * alpha_3d) + (roi * (1.0 - alpha_3d))
    else:
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_card[:, :, :3]

    return canvas


class InferenceDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform: Callable = None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        try:
            image = load_image(path)
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, path.stem


class MTGTrainDataset(Dataset):
    def __init__(self, image_paths: List[Path], label_map: Dict[str, int], transform: Callable, img_size: int = 224):
        self.image_paths = image_paths
        self.label_map = label_map
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        try:
            image = load_image(path)
            image = apply_random_background(image, self.img_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Warning: Error loading {path}: {e}")
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        label_id = self.label_map.get(path.stem, -1)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label_id


class MTGValidationDataset(Dataset):
    def __init__(self, image_paths: List[Path], label_map: Dict[str, int], img_size: int = 224):
        self.image_paths = image_paths
        self.label_map = label_map
        self.img_size = img_size
        self.transform_gallery = get_inference_transforms(img_size)
        self.transform_query = get_train_transforms(img_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path = self.image_paths[idx]
        try:
            raw_image = load_image(path)

            # clean gallery image
            if len(raw_image.shape) == 3 and raw_image.shape[2] == 4:
                rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            gallery_img = self.transform_gallery(image=rgb)["image"]

            # augmented query image
            playmat_bgr = apply_random_background(raw_image, self.img_size)
            playmat_rgb = cv2.cvtColor(playmat_bgr, cv2.COLOR_BGR2RGB)
            query_img = self.transform_query(image=playmat_rgb)["image"]

        except Exception as e:
            print(f"Validation error on {path}: {e}")
            gallery_img = torch.zeros((3, self.img_size, self.img_size))
            query_img = torch.zeros((3, self.img_size, self.img_size))

        label_id = self.label_map.get(path.stem, -1)

        return gallery_img, query_img, label_id


def visualize_augmentations(dataset: Dataset, output_path: str, num_images: int = 16) -> None:
    """Saves a grid of augmented images to disk for sanity checking."""
    indices = np.random.choice(len(dataset), num_images, replace=False)
    images = []

    for idx in indices:
        img_tensor, _ = dataset[idx]
        img = img_tensor.permute(1, 2, 0).cpu().numpy()

        # Un-normalize (ImageNet mean/std used by Albumentations)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean

        img = np.clip(img, 0, 1) * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

    grid_size = int(math.ceil(math.sqrt(num_images)))
    h, w, c = images[0].shape
    grid_img = np.zeros((grid_size * h, grid_size * w, c), dtype=np.uint8)

    for i, img in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        grid_img[row * h : (row + 1) * h, col * w : (col + 1) * w, :] = img

    cv2.imwrite(output_path, grid_img)
    print(f"Augmentation preview saved to: {output_path}")
