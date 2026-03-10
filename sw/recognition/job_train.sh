#!/usr/bin/env bash

#SBATCH --job-name=mtg_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # 1 main process
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --partition=h200
#SBATCH --mem=64G                   # RAM
#SBATCH --time=24:00:00             # time limit
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --mail-user=adamej14@fel.cvut.cz
#SBATCH --mail-type=END

ml Python/3.11.5-GCCcore-13.2.0
ml PyTorch/2.4.0-foss-2023b-CUDA-12.4.0
ml Albumentations/1.4.4-foss-2023b-CUDA-12.4.0
ml OpenCV/4.10.0-foss-2023b-CUDA-12.4.0-contrib
ml timm/0.6.13-foss-2023b-CUDA-12.4.0

IMG_DIR="/mnt/personal/adamej14/dataset"
SAVE_DIR="./check"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train.py \
    --input_dir="$IMG_DIR" \
    --save_dir="$SAVE_DIR" \
    --lr=3e-4 \
    --epochs=60
