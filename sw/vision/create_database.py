#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import MTGReconModel
from pathlib import Path

# --- CONFIG ---
MODEL_PATH = "/mnt/personal/adamej14/checkpoints/arcface_mtg_final.pth"
IMAGES_DIR = "/mnt/personal/adamej14/images/"
DATABASE_OUTPUT = "card_database.pth"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_inference_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def create_database():
    print(f"Loading model from {MODEL_PATH}...")
    
    all_files = list(Path(IMAGES_DIR).glob("*.png"))
    num_classes = len(all_files)
    
    model = MTGReconModel(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    transform = get_inference_transforms()
    
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
            img_tensor = aug.unsqueeze(0).to(DEVICE) # add batch dimension: [1, 3, 224, 224]
            
            embedding = model(img_tensor) # returns vector (512)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            vectors.append(embedding.cpu())
            names.append(file_path.stem)

    database_matrix = torch.cat(vectors) #[100000, 512]
    
    torch.save({
        "vectors": database_matrix,
        "names": names
    }, DATABASE_OUTPUT)
    
    print(f"Done! Database saved to {DATABASE_OUTPUT}")

if __name__ == "__main__":
    create_database()