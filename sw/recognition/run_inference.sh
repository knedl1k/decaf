#!/usr/bin/env bash

ml Python/3.13.5-GCCcore-14.3.0
ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
ml Albumentations/2.0.8-foss-2025b-CUDA-12.9.1
ml OpenCV/4.12.0-foss-2025b-CUDA-12.9.1-contrib
ml timm/1.0.25-foss-2025b-CUDA-12.9.1

MODEL_PATH="./check/arcface_mtg_final.pth"
DATABASE_PATH="card_database.pth"
TEST_IMG="../001.png"

python3 inference.py \
    --model="$MODEL_PATH" \
    --database="$DATABASE_PATH" \
    --img="$TEST_IMG"
