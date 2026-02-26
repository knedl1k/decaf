#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads model weights safely.
    Removes the ArcFace head weights from the checkpoint so that a model trained
    on a specific number of classes can be used for inference or fine-tuned on a new dataset.
    """
    print(f"Loading model weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # care only about the backbone and embedding layer for inference/transfer learning
    if "arcface.weight" in checkpoint:
        del checkpoint["arcface.weight"]

    try:
        model.load_state_dict(checkpoint, strict=False)
    except RuntimeError as e:
        print(f"Warning during loading weights: {e}")

    return model


def evaluate_metrics(
    model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device
) -> Dict[str, float]:
    """
    Evaluates the model on the validation set.
    Computes Cosine Similarity between gallery and query images, and calculates
    the False Match Rate (FMR) at a True Match Rate (TMR) of 95%.
    """
    model.eval()

    gallery_vecs = []
    query_vecs = []

    with torch.no_grad():
        for img_gallery, img_query, _ in val_loader:
            img_gallery = img_gallery.to(device)
            img_query = img_query.to(device)

            emb_g = model(img_gallery)
            emb_g = torch.nn.functional.normalize(emb_g, p=2, dim=1)

            emb_q = model(img_query)
            emb_q = torch.nn.functional.normalize(emb_q, p=2, dim=1)

            gallery_vecs.append(emb_g.cpu())
            query_vecs.append(emb_q.cpu())

    gallery_matrix = torch.cat(gallery_vecs)  # [N, 512]
    query_matrix = torch.cat(query_vecs)  # [N, 512]

    # similarity[i, j] = how much is query img 'i' similar to the gallery img 'j'
    similarity_matrix = torch.mm(query_matrix, gallery_matrix.t())  # [N, N]

    positives = similarity_matrix.diag()  # [N] Matches

    # mask out the diagonal to get non-matches (negatives)
    eye = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device).bool()
    negatives = similarity_matrix[~eye]  # [N * (N-1)]

    # True Match Rate
    target_tmr = 0.95
    pos_sorted, _ = torch.sort(positives)
    cutoff_index = int(len(positives) * (1 - target_tmr))
    threshold = pos_sorted[cutoff_index].item()

    # False Match Rate
    false_matches = (negatives > threshold).sum().item()
    num_negatives = negatives.numel()
    fmr = false_matches / num_negatives

    avg_pos = positives.mean().item()
    avg_neg = negatives.mean().item()

    return {
        "fmr_at_95_tmr": fmr,
        "threshold": threshold,
        "avg_pos_sim": avg_pos,
        "avg_neg_sim": avg_neg,
    }


def plot_training_curves(history: Dict[str, list], save_dir: str) -> None:
    """
    Generates and saves a 2x2 grid of training metrics plots.
    """
    try:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("MTG ArcFace Training Metrics", fontsize=16)

        # Plot 1: Training Loss
        axs[0, 0].plot(history["epoch"], history["loss"], "m-o", linewidth=2)
        axs[0, 0].set_title("Training Loss")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].grid(True)

        # Plot 2: FMR
        axs[0, 1].plot(history["epoch"], history["fmr"], "r-o", linewidth=2)
        axs[0, 1].set_yscale("log")
        axs[0, 1].set_title("Validation FMR @ 95% TMR")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("FMR (%)")
        axs[0, 1].grid(True, which="both", ls="--")

        # Plot 3: Similarities
        axs[1, 0].plot(history["epoch"], history["pos_sim"], "g-o", label="Avg Pos Sim", linewidth=2)
        axs[1, 0].plot(history["epoch"], history["neg_sim"], "r-o", label="Avg Neg Sim", linewidth=2)
        axs[1, 0].set_title("Cosine Similarity (Gallery vs Query)")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("Similarity (-1 to 1)")
        axs[1, 0].set_ylim(-0.2, 1.0)
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Plot 4: Threshold
        axs[1, 1].plot(history["epoch"], history["threshold"], "b-o", linewidth=2)
        axs[1, 1].set_title("Threshold for 95% TMR")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Threshold")
        axs[1, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Updated training curves saved to {plot_path}")
    except Exception as e:
        print(f"Warning: Failed to generate plots: {e}")
