#!/usr/bin/env bash

#SBATCH --job-name=mtg_inf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # 1 main process
#SBATCH --cpus-per-task=4           # 4 CPU cores for DataLoader
#SBATCH --gres=gpu:1                # 1 GPU card
#SBATCH --partition=h200fast
#SBATCH --mem=32G                   # RAM
#SBATCH --time=4:00:00              # time limit
#SBATCH --output=logs/inf_%j.out
#SBATCH --error=logs/inf_%j.err
#SBATCH --mail-user=adamej14@fel.cvut.cz
#SBATCH --mail-type=END

ml Python/3.13.5-GCCcore-14.3.0
ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
ml Albumentations/2.0.8-foss-2025b-CUDA-12.9.1
ml OpenCV/4.12.0-foss-2025b-CUDA-12.9.1-contrib
ml timm/1.0.25-foss-2025b-CUDA-12.9.1

MODEL_PATH="./check/arcface_mtg_ep60.pth"
DB_PATH="./card_database.pth"
DATASET_DIR="/mnt/personal/adamej14/dataset"
LABELED_DIR="$HOME/labeled"
DEBUG_DIR="check/eval_debug"

srun python3 evaluate_labeled.py \
    --model "$MODEL_PATH" \
    --database "$DB_PATH" \
    --ref_dir "$DATASET_DIR" \
    --real_dir "$LABELED_DIR" \
    --save_dir "check/eval_results" \
    --debug_dir "$DEBUG_DIR"
