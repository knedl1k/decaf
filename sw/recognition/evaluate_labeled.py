#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import shutil

from model import MTGReconModel
from data import get_inference_transforms, load_image
from utils import load_model_weights, detect_and_crop_card, parse_labeled_name


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ArcFace model on real MTG photos")
    parser.add_argument("--model", type=str, required=True, help="path to trained model checkpoint")
    parser.add_argument("--database", type=str, required=True, help="path to the index database")
    parser.add_argument("--real_dir", type=str, required=True, help="directory with real photos")
    parser.add_argument("--ref_dir", type=str, required=True, help="directory with reference digital images")
    parser.add_argument("--save_dir", type=str, default="./eval_results", help="directory to save results")
    parser.add_argument("--img_size", type=int, default=512, help="input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for inference")
    parser.add_argument("--debug_dir", type=str, default=None, help="dir to save debug contour images")
    return parser.parse_args()


def extract_features_from_real_photos(
    image_paths: List[Path],
    model: torch.nn.Module,
    transform,
    img_size: int,
    device: torch.device,
    debug_dir: str = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Performs on-the-fly homography extraction and inference.
    """
    model.eval()
    vectors = []
    valid_paths = []
    total_images = len(image_paths)

    print("Extracting features with on-the-fly cropping...")
    with torch.no_grad():
        for i, path in enumerate(image_paths):
            warped_card, debug_card = detect_and_crop_card(str(path), output_size=img_size)

            if debug_dir is not None and debug_card is not None:
                cv2.imwrite(os.path.join(debug_dir, path.name), debug_card)
                if warped_card is not None:
                    cv2.imwrite(os.path.join(debug_dir, "inputs", path.name), warped_card)

            if warped_card is None:
                print(f"Warning: Failed to crop {path.name}. Using raw image.")
                img = load_image(path)
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.cvtColor(warped_card, cv2.COLOR_BGR2RGB)

            img_tensor = transform(image=img)["image"].unsqueeze(0).to(device)

            emb = model(img_tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            vectors.append(emb.cpu())
            valid_paths.append(path)

            if (i + 1) % 10 == 0 or (i + 1) == total_images:
                print(f"Processed image {i + 1}/{total_images}.")

    return torch.cat(vectors), valid_paths


def generate_histogram(hit_sims: List[float], miss_sims: List[float], save_path: str):
    """Plots the distribution of cosine similarities."""
    plt.figure(figsize=(10, 6))
    if hit_sims:
        plt.hist(hit_sims, bins=np.linspace(0, 1, 40), alpha=0.6, color="green", label="Correct (Top-1)")
    if miss_sims:
        plt.hist(miss_sims, bins=np.linspace(0, 1, 40), alpha=0.6, color="red", label="Incorrect (Top-1)")
    plt.title("Distribution of Top-1 Cosine Similarities")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", alpha=0.5)
    plt.savefig(save_path)
    plt.close()


def compute_similarities_chunked(
    queries: torch.Tensor, db_vectors: torch.Tensor, chunk_size: int = 500
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes top-k similarities via chunked matrix multiplication.
    """
    top5_scores_list = []
    top5_idxs_list = []

    device = db_vectors.device

    for i in range(0, queries.shape[0], chunk_size):
        chunk = queries[i : i + chunk_size].to(device)
        sim_chunk = torch.mm(chunk, db_vectors.t())
        scores, idxs = torch.topk(sim_chunk, k=min(5, db_vectors.shape[0]), dim=1)

        top5_scores_list.append(scores.cpu())
        top5_idxs_list.append(idxs.cpu())

    return torch.cat(top5_scores_list, dim=0), torch.cat(top5_idxs_list, dim=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        os.makedirs(os.path.join(args.debug_dir, "correct"), exist_ok=True)
        os.makedirs(os.path.join(args.debug_dir, "incorrect"), exist_ok=True)

    print("Loading database...")
    db = torch.load(args.database, map_location=device, weights_only=False)
    db_vectors = db["vectors"].to(device)
    db_names = db["names"]

    print("Loading model...")
    model = MTGReconModel(num_classes=1).to(device)
    model = load_model_weights(model, args.model, device)

    all_files = [p for p in Path(args.real_dir).glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
    all_files.sort()

    if not all_files:
        print(f"No images found in {args.real_dir}")
        return

    transform = get_inference_transforms(args.img_size)

    # Feature extraction includes explicit alignment via detect_and_crop_card
    queries_cpu, valid_paths = extract_features_from_real_photos(
        all_files, model, transform, args.img_size, device, args.debug_dir
    )

    print("Calculating similarities...")
    top5_scores, top5_idxs = compute_similarities_chunked(queries_cpu, db_vectors)

    correct_1, correct_3, correct_5 = 0, 0, 0
    hit_sims, miss_sims = [], []
    correct_log, incorrect_log = [], []

    for i, path in enumerate(valid_paths):
        gt_name = parse_labeled_name(path.stem)
        preds = [db_names[idx.item()] for idx in top5_idxs[i]]
        scores = top5_scores[i].tolist()

        src = os.path.join(args.debug_dir, path.name)
        if gt_name == preds[0]:
            correct_1 += 1
            hit_sims.append(scores[0])
            correct_log.append(f"{path.name} -> {preds[0]} (Score: {scores[0]:.4f})")
            if args.debug_dir:
                dst = os.path.join(args.debug_dir, "correct", path.name)
                if os.path.exists(src):
                    shutil.move(src, dst)
        else:
            miss_sims.append(scores[0])
            incorrect_log.append(f"{path.name} -> Pred: {preds[0]} (Score: {scores[0]:.4f}) | GT: {gt_name}")
            if args.debug_dir:
                dst = os.path.join(args.debug_dir, "incorrect", path.name)
                if os.path.exists(src):
                    shutil.move(src, dst)

        if gt_name in preds[:3]:
            correct_3 += 1
        if gt_name in preds[:5]:
            correct_5 += 1

    total = len(valid_paths)

    save_reports(args.save_dir, correct_1, correct_3, correct_5, total, correct_log, incorrect_log)
    hist_path = os.path.join(args.save_dir, "confidence_histogram.png")
    generate_histogram(hit_sims, miss_sims, hist_path)

    print(f"\nEvaluation Finished. Top-1 Accuracy: {correct_1 / total * 100:.2f}%")
    print(f"Results saved to {args.save_dir}/")


def save_reports(save_dir: str, correct_1, correct_3, correct_5, total, correct_log, incorrect_log) -> None:
    report_path = os.path.join(save_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"--- Real Photo Evaluation Report ---\n")
        f.write(f"Total Images Evaluated: {total}\n\n")
        f.write(f"Top-1 Accuracy: {correct_1 / total * 100:.2f}% ({correct_1}/{total})\n")
        f.write(f"Top-3 Accuracy: {correct_3 / total * 100:.2f}% ({correct_3}/{total})\n")
        f.write(f"Top-5 Accuracy: {correct_5 / total * 100:.2f}% ({correct_5}/{total})\n\n")

    with open(os.path.join(save_dir, "correct_matches.txt"), "w") as f:
        f.write("\n".join(correct_log))

    with open(os.path.join(save_dir, "incorrect_matches.txt"), "w") as f:
        f.write("\n".join(incorrect_log))


if __name__ == "__main__":
    main()
