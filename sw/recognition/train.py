#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lr_scheduler import PolynomialLRWarmup
from model import MTGReconModel
from data import (
    MTGTrainDataset,
    MTGValidationDataset,
    get_train_transforms,
    get_inference_transforms,
    visualize_augmentations,
    InferenceDataset,
    RealValidationDataset,
)
from utils import evaluate_metrics, save_history, print_metrics, evaluate_real_domain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="distributed training for MTG card recognition")

    parser.add_argument("--ref_dir", type=str, required=True, help="directory with clean reference images")
    parser.add_argument("--real_val_dir", type=str, default=None, help="directory with real validation photos")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="directory to save models")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume training from")

    parser.add_argument("--batch_size", type=int, default=64, help="batch size per GPU")
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--img_size", type=int, default=512, help="input image size")

    parser.add_argument("--log_interval", type=int, default=100, help="batches to wait before logging")
    parser.add_argument("--num_workers", type=int, default=4, help="number of data loading workers")

    return parser.parse_args()


def update_history(history: Dict[str, list], metrics: Dict[str, Any], epoch: int, loss: float, lr: float) -> None:
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
    history["real_top1"].append(metrics["real_top1"])


def save_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, scheduler: Any, epoch: int, history: Dict[str, list], save_path: str
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
        },
        save_path,
    )


def load_checkpoint(
    resume_path: Optional[str],
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    is_master: bool,
) -> Tuple[int, Dict[str, list]]:
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
        "real_top1": [],
    }
    start_epoch = 0
    if resume_path and os.path.isfile(resume_path):
        if is_master:
            print(f"Loading checkpoint '{resume_path}'...")
        checkpoint = torch.load(resume_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.module.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            if is_master and "history" in checkpoint:
                history = checkpoint["history"]
            if is_master:
                print(f"Successfully resumed from epoch {start_epoch}")
    elif resume_path and is_master:
        print(f"Warning: Checkpoint '{resume_path}' not found. Starting from scratch.")

    return start_epoch, history


def prep_train_val(img_paths: list, val_size: int = 1000) -> Tuple[List[Path], List[Path]]:
    img_paths.sort()
    rng = np.random.RandomState(40)
    rng.shuffle(img_paths)
    return (img_paths[val_size:], img_paths[:val_size])


def setup_ddp() -> Tuple[int, torch.device, bool]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    is_master = dist.get_rank() == 0
    return local_rank, device, is_master


def prepare_data(args: argparse.Namespace, is_master: bool) -> Tuple[DataLoader, DataLoader, DistributedSampler, int]:
    train_paths, val_paths = prep_train_val(list(Path(args.ref_dir).glob("*.png")))
    # NN requires integer labels
    unique_names = sorted(list(set([p.stem for p in train_paths])))
    label_map = {name: i for i, name in enumerate(unique_names)}
    num_classes = len(unique_names)

    if is_master:
        print(f"Dataset split: {len(train_paths)} train cards, {len(val_paths)} val cards. Total: {num_classes}")

    train_dataset = MTGTrainDataset(
        img_paths=train_paths,
        label_map=label_map,
        transform=get_train_transforms(args.img_size),
        img_size=args.img_size,
    )

    if is_master:
        preview_path = os.path.join(args.save_dir, "preview_aug.png")
        try:
            visualize_augmentations(train_dataset, preview_path)
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

    val_dataset = MTGValidationDataset(img_paths=val_paths, label_map=label_map, img_size=args.img_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, sampler, num_classes


def init_training_regime(
    num_classes: int, args: argparse.Namespace, local_rank: int, device: torch.device, steps_per_epoch: int
) -> Tuple[nn.Module, nn.Module, optim.Optimizer, Any]:
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

    optimizer = optim.SGD(
        [{"params": backbone_params, "lr": args.lr * 0.1}, {"params": head_params, "lr": args.lr}],
        momentum=0.9,
        weight_decay=5e-4,
    )

    total_steps = args.epochs * steps_per_epoch
    warmup_epochs = 2
    warmup_iters = warmup_epochs * steps_per_epoch

    scheduler = PolynomialLRWarmup(optimizer, warmup_iters=warmup_iters, total_iters=total_steps, power=2.0)

    return model, criterion, optimizer, scheduler


def prepare_realEval_data(args: argparse.Namespace) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    if not args.real_val_dir:
        return None, None

    transform = get_inference_transforms(args.img_size)
    ref_files = list(Path(args.ref_dir).glob("*.png"))
    ref_dataset = InferenceDataset(ref_files, transform=transform)
    ref_loader = DataLoader(ref_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    real_files = [
        p for p in Path(args.real_val_dir).glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
    ]
    real_dataset = RealValidationDataset(real_files, transform=transform, img_size=args.img_size)
    print(f"DEBUG: loaded {len(real_dataset)} real photos.")
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return real_loader, ref_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    is_master: bool,
) -> float:
    model.train()
    running_loss = 0.0
    total_steps = len(dataloader)

    for i, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(imgs, labels)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if is_master and (i + 1) % args.log_interval == 0:
            print(f"Epoch [{epoch}/{args.epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}")

    return running_loss / total_steps


def main() -> None:
    args = parse_args()
    local_rank, device, is_master = setup_ddp()

    if is_master:
        print(f"Training started. World size: {dist.get_world_size()}")
        os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, sampler, num_classes = prepare_data(args, is_master)

    real_loader, ref_loader = prepare_realEval_data(args)

    model, criterion, optimizer, scheduler = init_training_regime(
        num_classes, args, local_rank, device, len(train_loader)
    )

    start_epoch, history = load_checkpoint(args.resume, model, optimizer, scheduler, device, is_master)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        sampler.set_epoch(epoch)

        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, args, is_master)

        if is_master:
            metrics = evaluate_metrics(model.module, val_loader, device)
            metrics["real_top1"] = None

            current_lr = optimizer.param_groups[1]["lr"]
            # plot_training_curves(history, args.save_dir)

            if epoch % 5 == 0 or epoch == args.epochs:
                save_path = os.path.join(args.save_dir, f"arcface_mtg_ep{epoch}.pth")
                save_checkpoint(model.module, optimizer, scheduler, epoch, history, save_path)
                print(f"Model saved to {save_path}")
                if real_loader and ref_loader:
                    metrics["real_top1"] = evaluate_real_domain(model.module, real_loader, ref_loader, device)

            print_metrics(metrics, epoch)
            update_history(history, metrics, epoch, avg_loss, current_lr)
            save_history(history, os.path.join(args.save_dir, "history.npy"))
        dist.barrier()

    if is_master:
        print(f"Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
