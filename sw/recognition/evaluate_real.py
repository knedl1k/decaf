#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from model import MTGReconModel
from data import InferenceDataset, get_inference_transforms, load_image
from utils import load_model_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ArcFace model on real MTG photos")
    parser.add_argument("--model", type=str, required=True, help="path to trained model checkpoint")
    parser.add_argument("--database", type=str, required=True, help="path to the index database")
    parser.add_argument("--real_dir", type=str, required=True, help="directory with real photos")
    parser.add_argument("--ref_dir", type=str, required=True, help="directory with reference digital images")
    parser.add_argument("--save_dir", type=str, default="./eval_results", help="directory to save results")
    parser.add_argument("--img_size", type=int, default=512, help="input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for faster inference")
    parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
    return parser.parse_args()


def find_ref_image(directory: Path, stem_name: str) -> str:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = directory / f"{stem_name}{ext}"
        if p.exists():
            return str(p)
    return None


def load_image_bgr(path: str, img_dim: int) -> np.ndarray:
    try:
        img = load_image(path)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            pass  # assumed BGR out of standard cv2.imread

        img = cv2.resize(img, (img_dim, img_dim))
        return img
    except Exception:
        return np.zeros((img_dim, img_dim, 3), dtype=np.uint8)


def create_error_grid(errors, ref_dir, save_path, max_errors=8, img_dim=256):
    if not errors:
        return
    np.random.shuffle(errors)
    errors = errors[:max_errors]

    # 3 columns: [Real Photo] | [Predicted Reference] |[True Reference]
    grid_img = np.zeros((len(errors) * img_dim, 3 * img_dim, 3), dtype=np.uint8)

    for i, (real_path, pred_name, gt_name) in enumerate(errors):
        real_img = load_image_bgr(real_path, img_dim)
        cv2.putText(real_img, "Real Photo", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(real_img, "Real Photo", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        pred_path = find_ref_image(Path(ref_dir), pred_name)
        pred_img = load_image_bgr(pred_path, img_dim) if pred_path else np.zeros((img_dim, img_dim, 3), dtype=np.uint8)
        cv2.putText(pred_img, f"Pred: {pred_name[:20]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(pred_img, f"Pred: {pred_name[:20]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        gt_path = find_ref_image(Path(ref_dir), gt_name)
        gt_img = load_image_bgr(gt_path, img_dim) if gt_path else np.zeros((img_dim, img_dim, 3), dtype=np.uint8)
        cv2.putText(gt_img, f"True: {gt_name[:20]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(gt_img, f"True: {gt_name[:20]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        row_y = i * img_dim
        grid_img[row_y : row_y + img_dim, 0:img_dim] = real_img
        grid_img[row_y : row_y + img_dim, img_dim : 2 * img_dim] = pred_img
        grid_img[row_y : row_y + img_dim, 2 * img_dim : 3 * img_dim] = gt_img

    cv2.imwrite(save_path, grid_img)


def create_success_grid(successes, save_path, grid_size=4, img_dim=256):
    if not successes:
        return
    np.random.shuffle(successes)
    successes = successes[: grid_size * grid_size]

    grid_img = np.zeros((grid_size * img_dim, grid_size * img_dim, 3), dtype=np.uint8)

    for i, (img_path, gt_name) in enumerate(successes):
        row, col = i // grid_size, i % grid_size
        img = load_image_bgr(img_path, img_dim)

        cv2.putText(img, gt_name[:20], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(img, gt_name[:20], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        grid_img[row * img_dim : (row + 1) * img_dim, col * img_dim : (col + 1) * img_dim] = img

    cv2.imwrite(save_path, grid_img)


def generate_histogram(hit_sims: List, miss_sims: List):
    plt.figure(figsize=(10, 6))
    plt.hist(hit_sims, bins=np.linspace(0, 1, 40), alpha=0.6, color="green", label="Correct (Top-1)")
    if miss_sims:
        plt.hist(miss_sims, bins=np.linspace(0, 1, 40), alpha=0.6, color="red", label="Incorrect (Top-1)")
    plt.title("Distribution of Top-1 Cosine Similarities")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", alpha=0.5)
    plt.savefig(os.path.join(args.save_dir, "confidence_histogram.png"))
    plt.close()


def generate_report(report_path: Path, c1: int, c3: int, c5: int, total: int, hit_sims: List, miss_sims: List):
    with open(report_path, "w") as f:
        f.write(f"--- Real Photo Evaluation Report ---\n")
        f.write(f"Total Images Evaluated: {total}\n\n")
        f.write(f"Top-1 Accuracy: {c1 / total * 100:.2f}% ({c1}/{total})\n")
        f.write(f"Top-3 Accuracy: {c3 / total * 100:.2f}% ({c3}/{total})\n")
        f.write(f"Top-5 Accuracy: {c5 / total * 100:.2f}% ({c5}/{total})\n\n")

        avg_hit = np.mean(hit_sims) if hit_sims else 0
        avg_miss = np.mean(miss_sims) if miss_sims else 0
        f.write(f"Average Confidence (Correct Match): {avg_hit:.4f}\n")
        f.write(f"Average Confidence (Incorrect Match): {avg_miss:.4f}\n")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading database...")
    db = torch.load(args.database, map_location=device, weights_only=False)
    db_vectors = db["vectors"].to(device)
    db_names = db["names"]

    print("Loading model...")
    model = MTGReconModel(num_classes=1).to(device)
    model = load_model_weights(model, args.model, device)
    model.eval()

    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG")
    all_files = []
    for ext in exts:
        all_files.extend(list(Path(args.real_dir).glob(ext)))
    all_files = list(set(all_files))
    all_files.sort()

    if not all_files:
        print(f"No images found in {args.real_dir}")
        return

    print(f"Found {len(all_files)} real photos to evaluate.")

    transform = get_inference_transforms(args.img_size)
    dataset = InferenceDataset(all_files, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Extracting features from real photos...")
    vectors = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            embeddings = model(imgs)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            vectors.append(embeddings)

    queries = torch.cat(vectors)  # [N_real, 512]

    print("Calculating similarities...")
    sims = torch.mm(queries, db_vectors.t())  # [N_real, N_db]

    top5_scores, top5_idxs = torch.topk(sims, k=min(5, len(db_names)), dim=1)

    correct_1, correct_3, correct_5 = 0, 0, 0
    hit_sims, miss_sims = [], []
    successes, errors = [], []

    for i in range(len(all_files)):
        gt_name = all_files[i].stem

        preds = [db_names[idx.item()] for idx in top5_idxs[i]]
        scores = top5_scores[i].tolist()

        if gt_name == preds[0]:
            correct_1 += 1
            hit_sims.append(scores[0])
            successes.append((str(all_files[i]), gt_name))
        else:
            miss_sims.append(scores[0])
            errors.append((str(all_files[i]), preds[0], gt_name))

        if gt_name in preds[:3]:
            correct_3 += 1
        if gt_name in preds[:5]:
            correct_5 += 1

    total = len(all_files)

    report_path = os.path.join(args.save_dir, "evaluation_report.txt")
    generate_report(report_path, correct_1, correct_3, correct_5, total)
    print(f"\nEvaluation Finished. Top-1 Accuracy: {correct_1 / total * 100:.2f}%")
    generate_histogram(hit_sims, miss_sims)
    create_success_grid(successes, os.path.join(args.save_dir, "success_grid.jpg"))
    create_error_grid(errors, args.ref_dir, os.path.join(args.save_dir, "error_grid.jpg"), max_errors=8)
    print(f"Results, logs, and visual grids saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
