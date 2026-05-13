#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import cv2
import torch
import argparse
import numpy as np
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any

from model import MTGReconModel
from data import get_inference_transforms, load_image
from utils import load_model_weights, smart_crop_card


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ArcFace model on pre-cropped real MTG photos")
    parser.add_argument("--model", type=str, required=True, help="path to trained model checkpoint")
    parser.add_argument("--database", type=str, required=True, help="path to the index database")
    parser.add_argument("--real_dir", type=str, required=True, help="directory with pre-cropped photos")
    parser.add_argument("--save_dir", type=str, default="./eval_results", help="directory to save results")
    parser.add_argument("--img_size", type=int, default=512, help="input image size")
    parser.add_argument("--debug_dir", type=str, default=None, help="dir to copy correctly/incorrectly matched images")
    return parser.parse_args()


def parse_mtg_filename(filename_stem: str) -> Tuple[str, str]:
    """
    Parses 'edition_collectorNumber_name-UUID' into (name, edition).
    Example: 'zen_21_kor-outfitter-00006596-1166-4a79-8443-ca9f82e6db4e' -> ('kor-outfitter', 'zen')
    """
    no_uuid = re.sub(
        r"-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", "", filename_stem
    )
    parts = no_uuid.split("_", 2)

    if len(parts) == 3:
        edition = parts[0]
        name = parts[2]
        return name, edition
    else:
        return no_uuid, "unknown"


def extract_features_from_real_photos(
    image_paths: List[Path],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    db_vectors: torch.Tensor,
    debug_dir: str = None,
) -> Tuple[torch.Tensor, List[Path]]:

    model.eval()
    vectors = []
    valid_paths = []
    total_images = len(image_paths)

    with torch.no_grad():
        for i, path in enumerate(image_paths):
            try:
                img = load_image(path)
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error: Failed to load {path.name}: {e}")
                continue

            cv2.imwrite(os.path.join(debug_dir, path.name), img)

            img_tensor_0 = transform(image=img)["image"].unsqueeze(0).to(device)
            img_tensor_180 = torch.flip(img_tensor_0, dims=[2, 3])
            batch = torch.cat([img_tensor_0, img_tensor_180], dim=0)
            emb = model(batch)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            sims = torch.mm(emb, db_vectors.t())
            max_scores, _ = torch.max(sims, dim=1)
            if max_scores[1] > max_scores[0]:
                vectors.append(emb[1:2].cpu())
            else:
                vectors.append(emb[0:1].cpu())

            valid_paths.append(path)

            if (i + 1) % 10 == 0 or (i + 1) == total_images:
                print(f"Processed image {i + 1}/{total_images}.")

    return torch.cat(vectors), valid_paths


def compute_similarities_chunked(
    queries: torch.Tensor, db_vectors: torch.Tensor, chunk_size: int = 500
) -> Tuple[torch.Tensor, torch.Tensor]:

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

    queries_cpu, valid_paths = extract_features_from_real_photos(
        all_files, model, transform, device, db_vectors, args.debug_dir
    )

    if not valid_paths:
        print("No valid features extracted. Exiting.")
        return

    print("Calculating similarities...")
    top5_scores, top5_idxs = compute_similarities_chunked(queries_cpu, db_vectors)

    correct_name = {1: 0, 3: 0, 5: 0}
    correct_edition = {1: 0, 3: 0, 5: 0}
    correct_exact = {1: 0, 3: 0, 5: 0}

    export_data = {
        "hit_sims_name": [],
        "miss_sims_name": [],
        "hit_sims_exact": [],
        "miss_sims_exact": [],
        "details": [],
    }

    correct_log, incorrect_log = [], []

    for i, path in enumerate(valid_paths):
        gt_name, gt_edition = parse_mtg_filename(path.stem)

        preds_parsed = [parse_mtg_filename(db_names[idx.item()]) for idx in top5_idxs[i]]
        pred_names = [p[0] for p in preds_parsed]
        pred_editions = [p[1] for p in preds_parsed]
        scores = top5_scores[i].tolist()

        match_name_1 = gt_name == pred_names[0]
        match_ed_1 = gt_edition == pred_editions[0]
        match_exact_1 = match_name_1 and match_ed_1

        if match_name_1:
            correct_name[1] += 1
            export_data["hit_sims_name"].append(scores[0])
        else:
            export_data["miss_sims_name"].append(scores[0])

        if match_exact_1:
            correct_exact[1] += 1
            export_data["hit_sims_exact"].append(scores[0])
        else:
            export_data["miss_sims_exact"].append(scores[0])

        if match_ed_1:
            correct_edition[1] += 1

        if gt_name in pred_names[:3]:
            correct_name[3] += 1
        if gt_edition in pred_editions[:3]:
            correct_edition[3] += 1
        if any(gt_name == n and gt_edition == e for n, e in zip(pred_names[:3], pred_editions[:3])):
            correct_exact[3] += 1

        if gt_name in pred_names[:5]:
            correct_name[5] += 1
        if gt_edition in pred_editions[:5]:
            correct_edition[5] += 1
        if any(gt_name == n and gt_edition == e for n, e in zip(pred_names[:5], pred_editions[:5])):
            correct_exact[5] += 1

        if match_exact_1:
            correct_log.append(f"{path.name} -> EXACT MATCH (Score: {scores[0]:.4f})")
        elif match_name_1:
            correct_log.append(
                f"{path.name} -> NAME MATCH ONLY | Pred: {pred_editions[0]} | GT: {gt_edition} (Score: {scores[0]:.4f})"
            )
        else:
            incorrect_log.append(
                f"{path.name} -> MISS | Pred: {pred_names[0]}_{pred_editions[0]} | GT: {gt_name}_{gt_edition} (Score: {scores[0]:.4f})"
            )

        export_data["details"].append(
            {
                "query": path.name,
                "gt_name": gt_name,
                "gt_edition": gt_edition,
                "pred_names": pred_names,
                "pred_editions": pred_editions,
                "scores": scores,
            }
        )

        if args.debug_dir:
            src = str(path)
            dst_folder = "correct" if match_name_1 else "incorrect"
            dst = os.path.join(args.debug_dir, dst_folder, path.name)
            shutil.copy(src, dst)

    total = len(valid_paths)

    export_data["metrics"] = {
        "total_images": total,
        "name": correct_name,
        "edition": correct_edition,
        "exact": correct_exact,
    }

    save_reports(args.save_dir, export_data, correct_log, incorrect_log)
    np.save(os.path.join(args.save_dir, "evaluation_data.npy"), export_data)

    print(f"\nEvaluation Finished. Analyzed {total} images.")
    print(f"Top-1 Name Match:  {correct_name[1] / total * 100:.2f}%")
    print(f"Top-1 Exact Match: {correct_exact[1] / total * 100:.2f}%")
    print(f"All raw data exported to: {os.path.join(args.save_dir, 'evaluation_data.npy')}")


def save_reports(save_dir: str, data: Dict[str, Any], correct_log: List[str], incorrect_log: List[str]) -> None:
    total = data["metrics"]["total_images"]
    name = data["metrics"]["name"]
    ed = data["metrics"]["edition"]
    exact = data["metrics"]["exact"]

    report_path = os.path.join(save_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"--- Real Photo Evaluation Report ---\n")
        f.write(f"Total Images Evaluated: {total}\n\n")

        f.write("--- NAME MATCH ---\n")
        f.write(f"Top-1: {name[1] / total * 100:.2f}% ({name[1]}/{total})\n")
        f.write(f"Top-3: {name[3] / total * 100:.2f}% ({name[3]}/{total})\n")
        f.write(f"Top-5: {name[5] / total * 100:.2f}% ({name[5]}/{total})\n\n")

        f.write("--- EXACT MATCH (Name + Edition) ---\n")
        f.write(f"Top-1: {exact[1] / total * 100:.2f}% ({exact[1]}/{total})\n")
        f.write(f"Top-3: {exact[3] / total * 100:.2f}% ({exact[3]}/{total})\n")
        f.write(f"Top-5: {exact[5] / total * 100:.2f}% ({exact[5]}/{total})\n\n")

        f.write("--- EDITION MATCH  ---\n")
        f.write(f"Top-1: {ed[1] / total * 100:.2f}% ({ed[1]}/{total})\n")

    with open(os.path.join(save_dir, "evaluation_log.txt"), "w") as f:
        f.write("=== CORRECT & PARTIAL MATCHES ===\n")
        f.write("\n".join(correct_log))
        f.write("\n\n=== INCORRECT MATCHES ===\n")
        f.write("\n".join(incorrect_log))


if __name__ == "__main__":
    main()
