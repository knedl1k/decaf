#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import MTGReconModel
from pathlib import Path
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="create an index of vectors with ArcFace NN")
    parser.add_argument("--model", type=str, required=True, help="path to directory with trained model")
    parser.add_argument("--images", type=str, required=True, help="path to directory with images")
    parser.add_argument("--save_dir", type=str, required=True, help="directory to save index")
    parser.add_argument("--img_size", type=int, default=224, help="input image size")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        stem = path.stem

        try:
            image = cv2.imread(str(path))
            if image is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, stem


def get_inference_transforms(img_size):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def create_database(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model}...")

    all_files = list(Path(args.images).glob("*.png"))
    all_files.sort()

    if not all_files:
        print("No images found!")
        return

    model = MTGReconModel(num_classes=1).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    if "arcface.weight" in checkpoint:
        del checkpoint["arcface.weight"]

    try:
        model.load_state_dict(checkpoint, strict=False)
    except RuntimeError as e:
        print(f"Warning during loading weights: {e}")

    model.eval()

    transform = get_inference_transforms(args.img_size)
    dataset = InferenceDataset(all_files, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    vectors = []
    names = []

    print(f"Starting inference on {len(dataset)} cards...")

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            for i, (imgs, batch_names) in enumerate(dataloader):
                imgs = imgs.to(device)

                embeddings = model(imgs)  # [Batch, 512]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                vectors.append(embeddings.cpu())
                names.extend(batch_names)

                if (i + 1) % 10 == 0:
                    print(f"Processed batch {i + 1}/{len(dataloader)}")

    database_matrix = torch.cat(vectors)
    torch.save({"vectors": database_matrix, "names": names}, args.save_dir)

    print(f"Done! Database saved to {args.save_dir}")


if __name__ == "__main__":
    args = parse_args()
    create_database(args)
