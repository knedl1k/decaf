#!/usr/bin/env bash

#SBATCH --job-name=mtg_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # 1 main process
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --partition=h200
#SBATCH --mem=64G                   # RAM
#SBATCH --time=24:00:00             # time limit
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-user=adamej14@fel.cvut.cz
#SBATCH --mail-type=END

ml Python/3.13.5-GCCcore-14.3.0
ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
ml Albumentations/2.0.8-foss-2025b-CUDA-12.9.1
ml OpenCV/4.12.0-foss-2025b-CUDA-12.9.1-contrib
ml timm/1.0.25-foss-2025b-CUDA-12.9.1

REF_DIR="/mnt/personal/adamej14/dataset"
REAL_DIR="/home/adamej14/labeled2"
SAVE_DIR="./check"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train.py \
    --ref_dir="$REF_DIR" \
    --real_val_dir="$REAL_DIR" \
    --save_dir="$SAVE_DIR" \
    --lr=0.1 \
    --epochs=60
