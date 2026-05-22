#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
from pathlib import Path

from model import MTGReconModel
from data import InferenceDataset, get_inference_transforms
from torch.utils.data import DataLoader


def strip_and_export_model(checkpoint_path: str, num_classes: int, img_size: int, output_onnx_path: str):
    """
    Strips optimizer/training states and exports the model to ONNX format.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    keys_to_remove = [k for k in state_dict.keys() if "arcface" in k]
    for k in keys_to_remove:
        del state_dict[k]

    print(f"Stripped {len(keys_to_remove)} ArcFace tensors. Instantiating model...")

    model = MTGReconModel(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    print(f"Exporting optimized graph to {output_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
    )

    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    onnx_size = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"Model compression complete: {original_size:.1f} MB -> {onnx_size:.1f} MB")


def compress_existing_database(input_path: str, output_path: str):
    """
    Loads an unoptimized PyTorch embedding database and
    compresses it into a half-precision (FP16) NumPy archive.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input database not found at {input_path}")

    print(f"Loading unoptimized database from {input_path}...")
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    vectors = data["vectors"]
    names = data["names"]

    vectors_np = vectors.numpy().astype(np.float16)
    names_np = np.array(names)

    print(f"Loaded {vectors_np.shape[0]} embeddings of dimension {vectors_np.shape[1]}.")

    print(f"Compressing and writing to {output_path}...")
    np.savez_compressed(output_path, embeddings=vectors_np, names=names_np)

    original_size = os.path.getsize(input_path) / (1024 * 1024)
    compressed_size = os.path.getsize(output_path) / (1024 * 1024)

    print("-" * 40)
    print("Compression Summary:")
    print(f"  Original PyTorch Size: {original_size:.2f} MB")
    print(f"  Compressed NumPy Size: {compressed_size:.2f} MB")
    print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MTG Recon model to edge-friendly format.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to your trained .pth file")
    parser.add_argument("--input_db", type=str, required=True, help="Path to your card database")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes the model was trained on")
    parser.add_argument("--img_size", type=int, default=512, help="Input image size")
    args = parser.parse_args()

    ONNX_OUTPUT = "mtg_recon_edge.onnx"
    DB_OUTPUT = "mtg_database.npz"

    strip_and_export_model(args.checkpoint, args.num_classes, args.img_size, ONNX_OUTPUT)
    compress_existing_database(args.ref_dir, DB_OUTPUT)
