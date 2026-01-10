#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import glob
import numpy as np
from pathlib import Path
import argparse
import math

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model import MTGReconModel # model.py

class MTGOnlineDataset(Dataset):
    def __init__(self, image_paths, label_map, transform=None, img_size=224):
        self.image_paths = image_paths
        self.label_map = label_map
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Image not loaded")
            
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)

            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.uint8)
        except Exception as e:
            # fallback for damaged files
            print(f"Warning: Error loading {path}: {e}")
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        file_stem = Path(path).stem
        label_id = self.label_map.get(file_stem, -1)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
            
        return image, label_id


def visualize_augmentations(dataset, output_path, num_images=16):
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    images = []
    for idx in indices:
        img_tensor, _ = dataset[idx]
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        
        img = np.clip(img, 0, 1) * 255
        img = img.astype(np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    
    grid_size = int(math.ceil(math.sqrt(num_images)))
    h, w, c = images[0].shape
    
    grid_img = np.zeros((grid_size * h, grid_size * w, c), dtype=np.uint8)
    
    for i, img in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        grid_img[row*h:(row+1)*h, col*w:(col+1)*w, :] = img
        
    cv2.imwrite(output_path, grid_img)
    print(f"Augmentation preview saved to: {output_path}")


# Augmentation Pipeline
def get_transforms(img_size):
    return A.Compose([
        # geometric deformations
        A.SafeRotate(limit=15.0, border_mode=cv2.BORDER_CONSTANT, p=0.7), 
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        
        # colors
        A.CoarseDropout(num_holes_range=(1,3), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # resize & padding
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
        
        # (0-255) -> (0.0-1.0) & normalizes according to the ImageNet standard
        A.Normalize(),
        # HWC (Height, Width, Channel) -> CHW (Channel, Height, Width)
        ToTensorV2()
    ])

def parse_args():
    parser = argparse.ArgumentParser(description="distributed training for MTG card recognition")
    
    parser.add_argument("--input_dir", type=str, required=True, help="path to directory with images")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="directory to save models")
    
    parser.add_argument("--batch_size", type=int, default=64, help="batch size per GPU")
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="input image size")
    
    parser.add_argument("--log_interval", type=int, default=100, help="how many batches to wait before logging")
    parser.add_argument("--num_workers", type=int, default=4, help="number of data loading workers")

    return parser.parse_args()

def main():
    args = parse_args()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device(f"cuda:{local_rank}")

    is_master = (dist.get_rank() == 0)
    if is_master:
        print(f"Training started. World size: {dist.get_world_size()}")
        print(f"Arguments: {args}")
        os.makedirs(args.save_dir, exist_ok=True)
    
    all_image_paths = list(Path(args.input_dir).glob("*.png"))
    if is_master:
        print(f"Found {len(all_image_paths)} unique cards.")

    all_image_paths = list(Path(args.input_dir).glob("*.png"))
    # NN needs numbers, not names (String -> Int)
    # 0, 1, 2...
    unique_names = sorted([p.stem for p in all_image_paths])
    label_map = {name: i for i, name in enumerate(unique_names)}
    num_classes = len(unique_names)

    if is_master:    
        print(f"No. of classes: {num_classes}")
    
    train_dataset = MTGOnlineDataset(
        image_paths = all_image_paths, 
        label_map = label_map, 
        transform = get_transforms(args.img_size),
        img_size = args.img_size
    )

    if is_master:
        print("Generating augment preview")
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
        sampler=sampler
    )

    model = MTGReconModel(num_classes=num_classes).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        
    if is_master:
        print("Start of training...")
    total_steps = len(train_loader)
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        if epoch < 3:
            if is_master:
                print(f"Epoch {epoch+1}: Backbone is FROZEN (Training head only)")
            if isinstance(model, (nn.DataParallel, DDP)):
                for param in model.module.backbone.parameters():
                    param.requires_grad = False
                for param in model.module.bn1.parameters():
                    param.requires_grad = False
            else:
                for param in model.backbone.parameters():
                    param.requires_grad = False
        else:
            if epoch == 3:
                if is_master:
                    print("Unfreezing backbone... Fine-tuning everything now.")
            
            if isinstance(model, (nn.DataParallel, DDP)):
                for param in model.module.backbone.parameters():
                    param.requires_grad = True
                for param in model.module.bn1.parameters():
                    param.requires_grad = True
            else:
                for param in model.backbone.parameters():
                    param.requires_grad = True

        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
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
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")
            
        avg_loss = running_loss / len(train_loader)
        if is_master:
            print(f"Epoch #{epoch+1} done. Average Loss: {avg_loss:.4f}")

            scheduler.step(avg_loss)
        
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(args.save_dir, f"arcface_mtg_ep{epoch+1}.pth")
                torch.save(model.module.state_dict(), save_path)
                print(f"Model saved to {save_path}")
    if is_master:
        print("Training complete.")
        torch.save(model.module.state_dict(), os.path.join(args.save_dir, "arcface_mtg_final.pth"))

    dist.destroy_process_group()
        
if __name__ == "__main__":
    main()
