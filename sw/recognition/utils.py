#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import cv2
import numpy as np
import torch


def parse_mtg_filename(filename_stem: str) -> tuple[str, str]:
    """
    Strips UUID and splits the filename into (name, edition).
    Handles formats like 'edition_collectorNumber_name', 'edition_name' and ignores UUIDs.
    Returns: (name, edition)
    """
    no_uuid = re.sub(
        r"-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", "", filename_stem
    )
    no_uuid = re.sub(r"(_\d+|\(\d+\)|\s+\d+)$", "", no_uuid).strip()

    parts = no_uuid.split("_", 2)

    if len(parts) >= 3:
        return parts[2], parts[0]
    elif len(parts) == 2:
        return parts[1], parts[0]
    else:
        return no_uuid, "unknown"


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def crop_card(
    image_path: str, output_width: int = 480, output_height: int = 670, debug: bool = False
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Detects a card in the image and warps it to a top-down view.
    Returns the warped image or None if no card is found, and debug image with contour
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error with loading {image_path}")
        return None, None

    # downscaling for faster edge detection
    ratio = img.shape[0] / 800.0
    orig = img.copy()
    img_resized = cv2.resize(img, (int(img.shape[1] / ratio), 800))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 75, 75)
    edges = cv2.Canny(blur, 30, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"Found no contours: {image_path}")
        return None, None

    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # largest_contour = contours[0]
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = np.float32(cv2.boxPoints(rect))

    box_orig = box * ratio
    rect_ordered = order_points(box_orig)

    dst_pts = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect_ordered, dst_pts)
    warped_img = cv2.warpPerspective(orig, matrix, (output_width, output_height))

    debug_img = None
    if debug:
        debug_img = img_resized.copy()
        cv2.drawContours(debug_img, [np.int32(box)], 0, (0, 0, 255), 3)

    return warped_img, debug_img


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads model weights safely.
    Removes the ArcFace head weights so a pre-trained model can be used for inference or fine-tuning.
    """
    print(f"Loading model weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if "arcface.weight" in state_dict:
        del state_dict["arcface.weight"]

    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"Warning during loading weights: {e}")

    return model


def evaluate_real_domain(
    model: torch.nn.Module,
    real_loader: torch.utils.data.DataLoader,
    ref_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Computes Top-1 accuracy on real photos."""
    model.eval()
    db_vectors, db_names = [], []

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for imgs, names in ref_loader:
            imgs = imgs.to(device)
            emb = torch.nn.functional.normalize(model(imgs), p=2, dim=1)
            db_vectors.append(emb.cpu())
            db_names.extend(names)

    db_matrix = torch.cat(db_vectors)  # [DB_Size, 512]
    correct_1, total = 0, 0

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for imgs, names in real_loader:
            imgs = imgs.to(device)
            emb = torch.nn.functional.normalize(model(imgs), p=2, dim=1)

            sims = torch.mm(emb.cpu(), db_matrix.t())
            best_idxs = torch.argmax(sims, dim=1)

            for i, gt_stem in enumerate(names):
                gt_name, gt_edition = parse_mtg_filename(gt_stem)
                pred_name, pred_edition = parse_mtg_filename(db_names[best_idxs[i].item()])
                if gt_name == pred_name and gt_edition == pred_edition:  # TODO: handle both separate
                    correct_1 += 1
                total += 1

    return (100.0 * correct_1 / total) if total > 0 else 0.0


def evaluate_metrics(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float | None]:
    """Computes Cosine Similarity and FMR at TMR 95%."""
    model.eval()
    gallery_vecs, query_vecs = [], []

    with torch.no_grad():
        for img_gallery, img_query, _ in val_loader:
            img_gallery, img_query = img_gallery.to(device), img_query.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                emb_g = torch.nn.functional.normalize(model(img_gallery), p=2, dim=1)
                emb_q = torch.nn.functional.normalize(model(img_query), p=2, dim=1)

            gallery_vecs.append(emb_g.cpu())
            query_vecs.append(emb_q.cpu())

    gallery_matrix = torch.cat(gallery_vecs)  # [N, 512]
    query_matrix = torch.cat(query_vecs)  # [N, 512]

    similarity_matrix = torch.mm(query_matrix, gallery_matrix.t())  # [N, N]
    positives = similarity_matrix.diag()

    # mask out the diagonal to get non-matches (negatives)
    eye = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device).bool()
    negatives = similarity_matrix[~eye]  # [N * (N-1)]

    target_tmr = 0.95
    pos_sorted, _ = torch.sort(positives)
    cutoff_index = int(len(positives) * (1 - target_tmr))
    threshold = pos_sorted[cutoff_index].item()

    fmr = (negatives > threshold).sum().item() / negatives.numel()

    top1_correct = (similarity_matrix.argmax(dim=1) == torch.arange(similarity_matrix.shape[0])).sum().item()
    top1_acc = top1_correct / similarity_matrix.shape[0]

    return {
        "fmr_at_95_tmr": fmr,
        "threshold": threshold,
        "avg_pos_sim": positives.mean().item(),
        "std_pos_sim": positives.std().item(),
        "avg_neg_sim": negatives.mean().item(),
        "std_neg_sim": negatives.std().item(),
        "top1_acc": top1_acc,
    }


def print_metrics(metrics: dict[str, float], epoch: int):
    print(f"--- VALIDATION RESULTS (Epoch {epoch}) ---")
    print(f"FMR @ TMR 95%: {metrics['fmr_at_95_tmr'] * 100:.4f} %")
    print(f"Top-1 (Synth): {metrics['top1_acc'] * 100:.2f} %")
    print(f"Top-1 (Real):  {metrics['real_top1']:.2f}%") if metrics["real_top1"] is not None else ""
    print(f"Threshold:     {metrics['threshold']:.4f}")
    print(f"Avg Pos Sim:   {metrics['avg_pos_sim']:.4f}")
    print(f"Avg Neg Sim:   {metrics['avg_neg_sim']:.4f}")
    print("-" * 42)


def save_history(history: dict[str, list], save_dir: str):
    np.save(save_dir, history)


def load_history(save_dir: str) -> dict[str, list]:
    return np.load(save_dir, allow_pickle=True).item()
