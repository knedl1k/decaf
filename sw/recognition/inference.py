#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import cv2
import argparse

from model import MTGReconModel
from data import load_image, get_inference_transforms
from utils import load_model_weights


def parse_args():
    parser = argparse.ArgumentParser(description="recognize a single MTG card image using ArcFace index")
    parser.add_argument("--img", type=str, required=True, help="path to the query image")
    parser.add_argument("--model", type=str, required=True, help="path to the trained model checkpoint")
    parser.add_argument("--database", type=str, required=True, help="path to the index database")
    parser.add_argument("--img_size", type=int, default=512, help="input image size")
    parser.add_argument("--num_candidates", type=int, default=3, help="number of top candidates to display")
    return parser.parse_args()


def recognize_card(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading database...")
    db = torch.load(args.database, map_location=device, weights_only=False)
    db_vectors = db["vectors"].to(device)
    db_names = db["names"]

    print("Loading model...")
    # initialize with 1 class (head weights are ignored during inference anyway)
    model = MTGReconModel(num_classes=1).to(device)
    model = load_model_weights(model, args.model, device)
    model.eval()

    try:
        img = load_image(args.img)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading test image: {e}")
        return

    transform = get_inference_transforms(args.img_size)
    aug = transform(image=img)["image"]
    img_tensor = aug.unsqueeze(0).to(device)

    with torch.no_grad():
        query_vector = model(img_tensor)
        query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=1)

    print(f"{query_vector[:10]=}")
    print(f"{db_vectors[0, :10]=}")
    print(f"{db_vectors[1, :10]=}")

    # Since the vectors are normalized, Cosine Similarity is just a scalar product (Dot Product).
    # Multiply the query vector (1, 512) by the database matrix (N, 512).T -> Result is (1, N)
    similarity_scores = torch.mm(query_vector, db_vectors.t())

    best_score, best_idx = torch.max(similarity_scores, dim=1)
    best_card_name = db_names[best_idx.item()]
    confidence = best_score.item() * 100

    print("\n" + "-" * 40)
    print(f"Test card:  {args.img}")
    print(f"Result:     {best_card_name}")
    print(f"Confidence: {confidence:.2f} %")
    print("-" * 40 + "\n")

    print(f"Top {args.num_candidates} candidates:")
    top_scores, top_idxs = torch.topk(similarity_scores, args.num_candidates)
    for i in range(args.num_candidates):
        idx = top_idxs[0][i].item()
        score = top_scores[0][i].item()
        print(f"{i + 1}. {db_names[idx]} ({score * 100:.2f}%)")


if __name__ == "__main__":
    args = parse_args()
    recognize_card(args)
