#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import MTGReconModel
import numpy as np

# --- CONFIG ---
MODEL_PATH = "/mnt/personal/adamej14/checkpoints/arcface_mtg_final.pth"
DATABASE_PATH = "card_database.pth"
TEST_IMAGE_PATH = "data/zen_21_kor-outfitter-00006596-1166-4a79-8443-ca9f82e6db4e.png"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_inference_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(),
        ToTensorV2()
    ])

def recognize_card():
    print("Loading database...")
    db = torch.load(DATABASE_PATH, map_location=DEVICE, weights_only=False)
    db_vectors = db["vectors"].to(DEVICE) # matrix [#classes, 512]
    db_names = db["names"]
    
    model = MTGReconModel(num_classes=1).to(DEVICE) # we do not care about num_classes
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if 'arcface.weight' in checkpoint:
        del checkpoint['arcface.weight']
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    img = cv2.imread(TEST_IMAGE_PATH)
    if img is None:
        print("Error: Test image not found.")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = get_inference_transforms()
    aug = transform(image=img)["image"]
    img_tensor = aug.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        query_vector = model(img_tensor)
        query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=1)

    print(f"{query_vector[:10]=}")
    print(f"{db_vectors[0, :10]=}")
    print(f"{db_vectors[1, :10]=}")
    
    # Since the vectors are normalized, Cosine Similarity is just a scalar product (Dot Product).
    # We multiply the card vector (1, 512) by the entire database matrix (512, 100000).
    # The result is a similarity score for each card in the database (-1 to 1).
    similarity_scores = torch.mm(query_vector, db_vectors.t())
    
    best_score, best_idx = torch.max(similarity_scores, dim=1)
    
    best_card_name = db_names[best_idx.item()]
    confidence = best_score.item() * 100
      
    print("-------------------------------")
    print(f"Test card:  {TEST_IMAGE_PATH}")
    print(f"Result:     {best_card_name}")
    print(f"Confidence: {confidence:.2f} %")
    print("--------------------------------")
    
    top_scores, top_idxs = torch.topk(similarity_scores, 3)
    print("TOP 3 candidates:")
    for i in range(3):
        idx = top_idxs[0][i].item()
        score = top_scores[0][i].item()
        print(f"{i+1}. {db_names[idx]} ({score*100:.2f}%)")

if __name__ == "__main__":
    recognize_card()
