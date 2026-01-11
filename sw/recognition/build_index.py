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

# --- CONFIG ---
# MODEL_PATH = "/mnt/personal/adamej14/checkpoints/arcface_mtg_final.pth"
# IMAGES_DIR = "/mnt/personal/adamej14/images/"
# DATABASE_OUTPUT = "card_database.pth"
# IMG_SIZE = 224
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="create an index of vectors with ArcFace NN")
    parser.add_argument("--model" , type=str, required=True, help="path to directory with trained model")
    parser.add_argument("--images", type=str, required=True, help="path to directory with images")
    parser.add_argument("--save_dir", type=str, required=True, help="directory to save index")
    parser.add_argument("--img_size", type=int, default=224, help="input image size")
    return parser.parse_args()

def get_inference_transforms(img_size):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(),
        ToTensorV2()
    ])

def create_database(args):
    print(f"Loading model from {args.model}...")
    
    all_files = list(Path(args.images).glob("*.png"))
    num_classes = len(all_files)
    
    model = MTGReconModel(num_classes=num_classes).to("cuda")
    model.load_state_dict(torch.load(args.model, map_location="cuda"))
    model.eval()

    transform = get_inference_transforms(args.img_size)
    
    vectors = []
    names = []

    print(f"Evaluating {len(all_files)} cards to database...")
    
    with torch.no_grad():
        for file_path in all_files:
            img = cv2.imread(str(file_path))
            if img is None: continue
            
            if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            aug = transform(image=img)["image"]
            img_tensor = aug.unsqueeze(0).to("cuda") # add batch dimension: [1, 3, 224, 224]
            
            embedding = model(img_tensor) # returns vector (512)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            vectors.append(embedding.cpu())
            names.append(file_path.stem)

    database_matrix = torch.cat(vectors) #[100000, 512]
    
    torch.save({
        "vectors": database_matrix,
        "names": names
    }, args.save_dir)
    
    print(f"Done! Database saved to {args.save_dir}")

if __name__ == "__main__":
    args = parse_args()
    create_database(args)
