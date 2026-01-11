#!/usr/bin/env bash

#SBATCH --job-name=mtg_recon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # 1 main process
#SBATCH --cpus-per-task=4           # 4 CPU cores for DataLoader
#SBATCH --gres=gpu:1                # 1 GPU card
#SBATCH --partition=gpu
#SBATCH --mem=32G                   # RAM
#SBATCH --time=24:00:00             # time limit
#SBATCH --output=recon_%j.out
#SBATCH --error=recon_%j.err
#SBATCH --mail-user=adamej14@fel.cvut.cz
#SBATCH --mail-type=END

ml Python/3.11.5-GCCcore-13.2.0
ml PyTorch/2.3.0-foss-2023b-CUDA-12.4.0
ml Albumentations/1.4.4-foss-2023b-CUDA-12.4.0
ml OpenCV/4.10.0-foss-2023b-CUDA-12.4.0-contrib
ml timm/0.6.13-foss-2023b-CUDA-12.4.0

MODEL_PATH="/mnt/personal/adamej14/checkpoints/arcface_mtg_final.pth"
IMG_DIR="/mnt/personal/adamej14/images"
OUT_PATH="./card_database.pth"

srun python3 build_index.py \
    --model "$MODEL_PATH" \
    --images "$IMG_DIR" \
    --save_dir "$OUT_PATH" \
    --batch_size 128 \
    --num_workers 4
