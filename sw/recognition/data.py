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
from collections.abc import Callable


def get_inference_transforms(img_size: int) -> A.Compose:
    """Transforms for validation gallery, database creation and single-image inference."""
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
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_REPLICATE,
                p=0.8,
            ),
            A.Perspective(scale=(0.02, 0.08), p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.2, 0.35),
                hole_width_range=(0.2, 0.35),
                fill="random",
                p=0.5,
            ),
            A.RandomSunFlare(src_radius=250, num_flare_circles_range=(2, 3), p=0.7),
            A.RandomShadow(num_shadows_limit=(1, 2), shadow_roi=(0, 0, 1, 1), p=0.4),
            A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, p=0.2),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.8),
            A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.5),
            A.ToGray(p=0.2),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.ImageCompression(quality_range=(50, 90), p=0.3),
            A.ISONoise(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def load_image(path: str | Path) -> np.ndarray:
    """Unified image loading to handle 16-bit, RGBA and standard BGR images."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {path}")

    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)

    return img


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Loads image and strictly converts to RGB, ignoring alpha if not needed."""
    img = load_image(path)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def apply_random_background(img: np.ndarray, target_size: int) -> np.ndarray:
    """Resizes the card and places it on a random noisy background, handling alpha channels."""
    h, w = img.shape[:2]
    scale = (target_size * 0.9) / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_card = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

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


class RealValidationDataset(Dataset):
    def __init__(self, img_paths: list[Path], transform: Callable, img_size: int) -> None:
        from utils import crop_card

        self.transform = transform
        self.img_size = img_size  # TODO: idk if it's still good idea to use it in crop_card
        self.valid_data = []

        print(f"Pre-computing homography crops for {len(img_paths)} real images...")
        for path in img_paths:
            warped, _ = crop_card(image_path=str(path))
            if warped is not None:
                rgb_img = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                self.valid_data.append((rgb_img, path.stem))

    def __len__(self) -> int:
        return len(self.valid_data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        img, stem = self.valid_data[index]
        if self.transform:
            tensor = self.transform(image=img)["image"].contiguous()
        return tensor, stem


class InferenceDataset(Dataset):
    def __init__(self, img_paths: list[Path], transform: Callable | None = None) -> None:
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor | np.ndarray, str]:
        path = self.img_paths[index]
        try:
            img = load_rgb_image(path)
        except Exception as e:
            print(f"{type(self).__name__}: Error loading {path}: {e}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(image=img)["image"].contiguous()
        return img, path.stem


class MTGTrainDataset(Dataset):
    def __init__(self, img_paths: list[Path], label_map: dict[str, int], transform: Callable, img_size: int) -> None:
        self.img_paths = img_paths
        self.label_map = label_map
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor | np.ndarray, int]:
        path = self.img_paths[index]
        try:
            img = load_image(path)
            img = apply_random_background(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"{type(self).__name__}: Error loading {path}: {e}")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        label_id = self.label_map.get(path.stem, -1)

        if self.transform:
            img = self.transform(image=img)["image"].contiguous()
        return img, label_id


class MTGValidationDataset(Dataset):
    def __init__(self, img_paths: list[Path], label_map: dict[str, int], img_size: int) -> None:
        self.img_paths = img_paths
        self.label_map = label_map
        self.img_size = img_size
        self.transform_gallery = get_inference_transforms(img_size)
        self.transform_query = get_train_transforms(img_size)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        path = self.img_paths[index]
        try:
            raw_img = load_image(path)

            # clean gallery image
            if len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
                rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            gallery_img = self.transform_gallery(image=rgb)["image"].contiguous()

            # augmented query image
            playmat_bgr = apply_random_background(raw_img, self.img_size)
            playmat_rgb = cv2.cvtColor(playmat_bgr, cv2.COLOR_BGR2RGB)
            query_img = self.transform_query(image=playmat_rgb)["image"].contiguous()
        except Exception as e:
            print(f"{type(self).__name__}: Validation error on {path}: {e}")
            gallery_img = torch.zeros((3, self.img_size, self.img_size))
            query_img = torch.zeros((3, self.img_size, self.img_size))

        label_id = self.label_map.get(path.stem, -1)
        return gallery_img, query_img, label_id


def visualize_augmentations(dataset: Dataset, output_path: str, num_imgs: int = 16) -> None:
    """Saves a grid of augmented images to disk for sanity checking."""
    indices = np.random.choice(len(dataset), num_imgs, replace=False)
    imgs = []

    for idx in indices:
        img_tensor, _ = dataset[idx]
        img = img_tensor.permute(1, 2, 0).cpu().numpy()

        # Un-normalize (mean/std used by Albumentations)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean

        img = np.clip(img, 0, 1) * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs.append(img)

    grid_size = int(math.ceil(math.sqrt(num_imgs)))
    h, w, c = imgs[0].shape
    grid_img = np.zeros((grid_size * h, grid_size * w, c), dtype=np.uint8)

    for i, img in enumerate(imgs):
        row = i // grid_size
        col = i % grid_size
        grid_img[row * h : (row + 1) * h, col * w : (col + 1) * w, :] = img

    cv2.imwrite(output_path, grid_img)
    print(f"Augmentation preview saved to: {output_path}")
