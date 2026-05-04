#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from model import MTGReconModel
from data import InferenceDataset, get_inference_transforms
from utils import load_model_weights


def parse_args():
    parser = argparse.ArgumentParser(description="create an index of vectors with ArcFace NN")
    parser.add_argument("--model", type=str, required=True, help="path to trained model checkpoint")
    parser.add_argument("--images", type=str, required=True, help="path to directory with images")
    parser.add_argument("--save_dir", type=str, required=True, help="path to save the output index database")
    parser.add_argument("--img_size", type=int, default=512, help="input image size")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
    return parser.parse_args()


def create_database(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_files = list(Path(args.images).glob("*.png"))
    all_files.sort()

    if not all_files:
        print(f"No images found in {args.images}!")
        return

    print(f"Found {len(all_files)} images. Preparing to load model...")

    # initialize model with 1 dummy class since we only need the backbone to extract embeddings
    model = MTGReconModel(num_classes=1).to(device)
    model = load_model_weights(model, args.model, device)
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
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            for i, (imgs, batch_names) in enumerate(dataloader):
                imgs = imgs.to(device)

                embeddings = model(imgs)  # [Batch, 512]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                vectors.append(embeddings.cpu())
                names.extend(batch_names)

                if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                    print(f"Processed batch {i + 1}/{len(dataloader)}")

    database_matrix = torch.cat(vectors)
    torch.save({"vectors": database_matrix, "names": names}, args.save_dir)

    print(f"Done! Database saved to {args.save_dir}")


if __name__ == "__main__":
    args = parse_args()
    create_database(args)
