#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort
import os

from utils import crop_card, parse_mtg_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust Edge Inference for MTG cards on Raspberry Pi")
    parser.add_argument("--img", type=str, required=True, help="Path to the captured query image")
    parser.add_argument("--model", type=str, required=True, help="Path to mtg_recon_edge.onnx")
    parser.add_argument("--database", type=str, required=True, help="Path to mtg_database_edge.npz")
    parser.add_argument("--img_size", type=int, default=512, help="Input image size for the model")
    parser.add_argument("--num_candidates", type=int, default=3, help="Number of top candidates to display")
    return parser.parse_args()


def preprocess_image(img: np.ndarray, img_size: int) -> np.ndarray:
    """Replicates Albumentations normalization for ONNX."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    scale = img_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = (img_size - new_h) // 2
    bottom = img_size - new_h - top
    left = (img_size - new_w) // 2
    right = img_size - new_w - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    img_float = img_padded.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std

    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    return np.expand_dims(img_transposed, axis=0)


def extract_normalized_vector(ort_session: ort.InferenceSession, tensor: np.ndarray) -> np.ndarray:
    """Runs ONNX inference and L2-normalizes the output."""
    vec = ort_session.run(None, {"input": tensor})[0][0]
    return vec / np.linalg.norm(vec)


def main() -> None:
    args = parse_args()

    print("Loading FP16 Database...")
    t0 = time.time()
    db = np.load(args.database)
    db_vectors = db["embeddings"].astype(np.float32)
    db_names = db["names"]
    print(f"Database loaded in {time.time() - t0:.2f}s")

    print("Initializing ONNX Runtime...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(args.model, sess_options=session_options, providers=["CPUExecutionProvider"])

    cropped_img_0, _ = crop_card(image_path=args.img)
    if cropped_img_0 is None:
        print("Could not find a card in the image! Exiting.")
        return

    cropped_img_180 = cv2.rotate(cropped_img_0, cv2.ROTATE_180)
    cv2.imwrite(os.path.join(".", "crop.jpg"), cropped_img_180)

    tensor_0 = preprocess_image(cropped_img_0, args.img_size)
    tensor_180 = preprocess_image(cropped_img_180, args.img_size)

    print("Running Inference (0° and 180°)...")
    t1 = time.time()
    vec_0 = extract_normalized_vector(ort_session, tensor_0)
    vec_180 = extract_normalized_vector(ort_session, tensor_180)
    print(f"Inference complete in {time.time() - t1:.2f}s")

    sims_0 = np.dot(db_vectors, vec_0)
    sims_180 = np.dot(db_vectors, vec_180)

    if np.max(sims_180) > np.max(sims_0):
        best_sims = sims_180
        best_orientation = "180° (Flipped)"
    else:
        best_sims = sims_0
        best_orientation = "0° (Standard)"

    best_idxs = np.argsort(best_sims)[::-1][: args.num_candidates]

    top1_name, top1_edition = parse_mtg_filename(db_names[best_idxs[0]])

    print("\n" + "=" * 50)
    print(f"Test card:   {args.img}")
    print(f"Orientation: {best_orientation}")
    print(f"Result:      {top1_name} [{top1_edition.upper()}]")
    print(f"Confidence:  {best_sims[best_idxs[0]] * 100:.2f} %")
    print("=" * 50 + "\n")

    print(f"Top {args.num_candidates} Candidates:")
    for i, idx in enumerate(best_idxs):
        name, ed = parse_mtg_filename(db_names[idx])
        print(f"  {i + 1}. {name} [{ed.upper()}] - {best_sims[idx] * 100:.2f}%")


if __name__ == "__main__":
    main()
