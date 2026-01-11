#!/usr/bin/env bash
ml Python/3.11.5-GCCcore-13.2.0
ml PyTorch/2.4.0-foss-2023b-CUDA-12.4.0
ml Albumentations/1.4.4-foss-2023b-CUDA-12.4.0
ml OpenCV/4.10.0-foss-2023b-CUDA-12.4.0-contrib
ml timm/0.6.13-foss-2023b-CUDA-12.4.0

MODEL_PATH="./check/arcface_mtg_final.pth"
DATABASE_PATH="card_database.pth"

python3 inference.py \
    --model="$MODEL_PATH" \
    --database="$DATABASE_PATH"
