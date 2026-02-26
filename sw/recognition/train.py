#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import MTGReconModel
from data import MTGTrainDataset, MTGValidationDataset, get_train_transforms, visualize_augmentations
from utils import evaluate_metrics, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="distributed training for MTG card recognition")

    parser.add_argument("--input_dir", type=str, required=True, help="path to directory with images")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="directory to save models")

    parser.add_argument("--batch_size", type=int, default=64, help="batch size per GPU")
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="input image size")

    parser.add_argument("--log_interval", type=int, default=100, help="how many batches to wait before logging")
    parser.add_argument("--num_workers", type=int, default=4, help="number of data loading workers")

    return parser.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    is_master = dist.get_rank() == 0

    if is_master:
        print(f"Training started. World size: {dist.get_world_size()}")
        os.makedirs(args.save_dir, exist_ok=True)

    all_image_paths = list(Path(args.input_dir).glob("*.png"))
    all_image_paths.sort()
    rng = np.random.RandomState(42)
    rng.shuffle(all_image_paths)

    VAL_SIZE = 1000
    train_paths = all_image_paths[VAL_SIZE:]
    val_paths = all_image_paths[:VAL_SIZE]

    # NN requires integer labels
    unique_names = sorted(list(set([p.stem for p in train_paths])))
    label_map = {name: i for i, name in enumerate(unique_names)}
    num_classes = len(unique_names)

    if is_master:
        print(f"Dataset split: {len(train_paths)} train cards, {len(val_paths)} val cards. Total: {num_classes}")

    # Datasets & Loaders
    train_dataset = MTGTrainDataset(
        image_paths=train_paths,
        label_map=label_map,
        transform=get_train_transforms(args.img_size),
        img_size=args.img_size,
    )

    if is_master:
        preview_path = os.path.join(args.save_dir, "preview_aug.png")
        try:
            visualize_augmentations(train_dataset, preview_path, num_images=16)
        except Exception as e:
            print(f"Warning: Failed to generate aug visualization: {e}")

    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    val_dataset = MTGValidationDataset(image_paths=val_paths, label_map=label_map, img_size=args.img_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    with open("vals.txt", "w") as val_txt:
        print(f"{val_dataset.label_map}", file=val_txt)

    model = MTGReconModel(num_classes=num_classes).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Optimizer & Scheduler Setup
    criterion = nn.CrossEntropyLoss()

    backbone_params = list(model.module.backbone.parameters())
    head_params = (
        list(model.module.bn1.parameters())
        + list(model.module.fc.parameters())
        + list(model.module.bn2.parameters())
        + list(model.module.arcface.parameters())
    )

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # TODO: https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
    optimizer = optim.AdamW(
        [{"params": backbone_params, "lr": args.lr * 0.1}, {"params": head_params, "lr": args.lr}], weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    if is_master:
        print("Start of training...")
        history = {"epoch": [], "loss": [], "fmr": [], "threshold": [], "pos_sim": [], "neg_sim": []}

    total_steps = len(train_loader)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # TODO: should be useless, but getting some weird errors, so this stays here for now
            if (labels < 0).any() or (labels >= num_classes).any():
                raise ValueError(
                    f"Label out of bounds! Min: {labels.min()}, Max: {labels.max()}, Classes: {num_classes}"
                )

            # forward pass
            outputs = model(images, labels)
            loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()

            if is_master and (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}")

        metric_tensors = torch.zeros(1, device=device)

        if is_master:
            metrics = evaluate_metrics(model.module, val_loader, device)

            print(f"--- VALIDATION RESULTS (Epoch {epoch + 1}) ---")
            print(f"FMR @ TMR 95%: {metrics['fmr_at_95_tmr'] * 100:.4f} %")
            print(f"Threshold:     {metrics['threshold']:.4f}")
            print(f"Avg Pos Sim:   {metrics['avg_pos_sim']:.4f}")
            print(f"Avg Neg Sim:   {metrics['avg_neg_sim']:.4f}")
            print("-" * 42)

            metric_tensors[0] = metrics["fmr_at_95_tmr"]

            history["epoch"].append(epoch + 1)
            history["loss"].append(running_loss / len(train_loader))
            history["fmr"].append(metrics["fmr_at_95_tmr"] * 100)
            history["threshold"].append(metrics["threshold"])
            history["pos_sim"].append(metrics["avg_pos_sim"])
            history["neg_sim"].append(metrics["avg_neg_sim"])

            plot_training_curves(history, args.save_dir)

            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(args.save_dir, f"arcface_mtg_ep{epoch + 1}.pth")
                torch.save(model.module.state_dict(), save_path)
                print(f"Model saved to {save_path}")

        # sync metrics across all GPUs for the scheduler
        dist.broadcast(metric_tensors, src=0)
        scheduler.step(metric_tensors.item())
        dist.barrier()

    if is_master:
        print("Training complete.")
        torch.save(model.module.state_dict(), os.path.join(args.save_dir, "arcface_mtg_final.pth"))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
