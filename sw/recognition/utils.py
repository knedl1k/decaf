#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
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

    state_dict = checkpoint["model_state_dict"]

    # care only about the backbone and embedding layer for inference/transfer learning
    if "arcface.weight" in state_dict:
        del state_dict["arcface.weight"]

    try:
        model.load_state_dict(state_dict, strict=False)
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

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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

    fmr = (negatives > threshold).sum().item() / negatives.numel()

    # Top-1 accuracy
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


def print_metrics(metrics: Dict[str, float], epoch: int):
    print(f"--- VALIDATION RESULTS (Epoch {epoch}) ---")
    print(f"FMR @ TMR 95%: {metrics['fmr_at_95_tmr'] * 100:.4f} %")
    print(f"Threshold:     {metrics['threshold']:.4f}")
    print(f"Avg Pos Sim:   {metrics['avg_pos_sim']:.4f}")
    print(f"Avg Neg Sim:   {metrics['avg_neg_sim']:.4f}")
    print("-" * 42)


def save_history(history: Dict[str, list], save_dir: str) -> None:
    np.save(save_dir, history)


def load_history(save_dir) -> Dict[str, list]:
    return np.load(save_dir, allow_pickle=True).item()


def plot_training_curves(history: Dict[str, list], save_dir: str) -> None:
    """
    Generates and saves a 2x2 grid of training metrics plots.
    """
    try:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Training Metrics", fontsize=16)
        epochs = history["epoch"]

        ax1 = axs[0, 0]
        ax1.plot(epochs, history["loss"], "m-o", linewidth=2, label="Loss")
        ax1.set_title("Training Loss & Learning Rate")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)
        ax1_lr = ax1.twinx()
        ax1_lr.plot(epochs, history["lr"], "k--", linewidth=2, alpha=0.7, label="Learning Rate")
        ax1_lr.set_ylabel("Learning Rate")
        ax1_lr.set_yscale("log")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_lr.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")

        ax2 = axs[0, 1]
        fmr = np.array(history["fmr"]) * 100
        ax2.plot(epochs, fmr, "r-o", linewidth=2)
        ax2.set_yscale("log")
        ax2.set_title("Validation FMR @ 95% TMR")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("FMR (%)")
        ax2.grid(True, which="both", ls="--")

        ax3 = axs[1, 0]
        pos = np.array(history["pos_sim"])
        pos_std = np.array(history.get("std_pos_sim", [0] * len(pos)))
        neg = np.array(history["neg_sim"])
        neg_std = np.array(history.get("std_neg_sim", [0] * len(neg)))
        thresh = np.array(history["threshold"])
        ax3.plot(epochs, pos, "g-o", label="Avg Pos Sim", linewidth=2)
        ax3.fill_between(epochs, pos - pos_std, pos + pos_std, color="green", alpha=0.2)
        ax3.plot(epochs, neg, "g-o", label="Avg Neg Sim", linewidth=2)
        ax3.fill_between(epochs, neg - neg_std, neg + neg_std, color="green", alpha=0.2)
        ax3.plot(epochs, thresh, "b--", label="Threshold (95 % TMR)", linewidth=2)
        ax3.set_title("Cosine Similarities & Threshold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Similarity (-1 to 1)")
        ax3.set_ylim(-0.2, 1.0)
        ax3.legend(loc="center right")
        ax3.grid(True)

        ax4 = axs[1, 1]
        top1 = np.array(history["top1_acc"]) * 100
        ax4.plot(epochs, top1, "c-o", linewidth=2)
        ax4.set_title("Top-1 Validation Accuracy")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Accuracy (%)")
        ax4.set_ylim(max(0, min(top1) - 5), 100.5)
        ax4.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Updated training curves saved to {plot_path}")
    except Exception as e:
        print(f"Warning: Failed to generate plots: {e}")
