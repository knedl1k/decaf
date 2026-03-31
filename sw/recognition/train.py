#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from model import MTGReconModel
from data import MTGTrainDataset, MTGValidationDataset, get_train_transforms, visualize_augmentations
from utils import evaluate_metrics, save_history, plot_training_curves, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="distributed training for MTG card recognition")

    parser.add_argument("--input_dir", type=str, required=True, help="path to directory with images")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="directory to save models")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume training from")

    parser.add_argument("--batch_size", type=int, default=64, help="batch size per GPU")
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--img_size", type=int, default=512, help="input image size")

    parser.add_argument("--log_interval", type=int, default=100, help="how many batches to wait before logging")
    parser.add_argument("--num_workers", type=int, default=4, help="number of data loading workers")

    return parser.parse_args()


def update_history(history: Dict[str, list], metrics: Dict[str, float], epoch: int, loss: float, lr: float):
    history["epoch"].append(epoch)
    history["loss"].append(loss)
    history["lr"].append(lr)
    history["fmr"].append(metrics["fmr_at_95_tmr"])
    history["threshold"].append(metrics["threshold"])
    history["pos_sim"].append(metrics["avg_pos_sim"])
    history["std_pos_sim"].append(metrics["std_pos_sim"])
    history["neg_sim"].append(metrics["avg_neg_sim"])
    history["std_neg_sim"].append(metrics["std_neg_sim"])
    history["top1_acc"].append(metrics["top1_acc"])


def save_model(
    mod_st: Dict[str, torch.Tensor],
    opt_st: Dict[str, Any],
    sch_st: Dict[str, Any],
    epoch: int,
    history: Dict[str, list],
    save_path: str,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": mod_st,
            "optimizer_state_dict": opt_st,
            "scheduler_state_dict": sch_st,
            "history": history,
        },
        save_path,
    )


def main():
    args = parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
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

    criterion = nn.CrossEntropyLoss()

    backbone_params = list(model.module.backbone.parameters())
    head_params = (
        list(model.module.bn1.parameters())
        + list(model.module.fc.parameters())
        + list(model.module.bn2.parameters())
        + list(model.module.arcface.parameters())
    )

    optimizer = optim.AdamW(
        [{"params": backbone_params, "lr": args.lr * 0.1}, {"params": head_params, "lr": args.lr}], weight_decay=1e-4
    )

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch

    warmup_epochs = 2
    warmup_iters = warmup_epochs * steps_per_epoch

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_iters), eta_min=1e-6)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])

    start_epoch = 0
    if is_master:
        print("Start of training...")
        history = {
            "epoch": [],
            "loss": [],
            "lr": [],
            "fmr": [],
            "threshold": [],
            "pos_sim": [],
            "std_pos_sim": [],
            "neg_sim": [],
            "std_neg_sim": [],
            "top1_acc": [],
        }
    else:
        history = {}

    if args.resume and os.path.isfile(args.resume):
        if is_master:
            print(f"Loading checkpoint '{args.resume}'...")
        checkpoint = torch.load(args.resume, map_location=device)

        if "model_state_dict" in checkpoint:
            model.module.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            if is_master and "history" in checkpoint:
                history = checkpoint["history"]
            if is_master:
                print(f"Successfully resumed from epoch {start_epoch}")
    elif args.resume and is_master:
        print(f"Warning: Checkpoint '{args.resume}' not found. Starting from scratch.")

    total_steps = len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images, labels)
                loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if is_master and (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}")

        # metric_tensors = torch.zeros(1, device=device)

        if is_master:
            metrics = evaluate_metrics(model.module, val_loader, device)
            print_metrics(metrics, epoch + 1)
            # metric_tensors[0] = metrics["fmr_at_95_tmr"]
            current_lr = optimizer.param_groups[1]["lr"]
            update_history(history, metrics, epoch + 1, running_loss / len(train_loader), current_lr)
            save_history(history, f"{args.save_dir}/history.npy")
            # plot_training_curves(history, args.save_dir)

            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(args.save_dir, f"arcface_mtg_ep{epoch + 1}.pth")
                save_model(
                    model.module.state_dict(),
                    optimizer.state_dict(),
                    scheduler.state_dict(),
                    epoch + 1,
                    history,
                    save_path,
                )
                print(f"Model saved to {save_path}")

        # sync metrics across all GPUs for the scheduler
        # dist.broadcast(metric_tensors, src=0)
        # scheduler.step(metric_tensors.item())
        dist.barrier()

    if is_master:
        save_path = os.path.join(args.save_dir, f"arcface_mtg_final.pth")
        save_model(
            model.module.state_dict(), optimizer.state_dict(), scheduler.state_dict(), args.epochs, history, save_path
        )
        print(f"Training complete, model saved to '{save_path}''.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
