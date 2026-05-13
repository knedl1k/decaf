#!/usr/bin/env bash

#SBATCH --job-name=mtg_inf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpufast
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/inf_%j.out
#SBATCH --error=logs/inf_%j.err
#SBATCH --mail-user=adamej14@fel.cvut.cz
#SBATCH --mail-type=END

ml Python/3.13.5-GCCcore-14.3.0
ml PyTorch/2.10.0-foss-2025b
ml Albumentations/2.0.8-foss-2025b
ml OpenCV/4.12.0-foss-2025b-contrib
ml timm/1.0.25-foss-2025b

MODEL_PATH="./check/arcface_mtg_ep60.pth"
DB_PATH="./card_database.pth"
DATASET_DIR="/mnt/personal/adamej14/dataset"
LABELED_DIR="$HOME/labeled2"
DEBUG_DIR="check/eval_debug"

rm -rf $DEBUG_DIR

srun python3 evaluate_labeled.py \
    --model "$MODEL_PATH" \
    --database "$DB_PATH" \
    --real_dir "$LABELED_DIR" \
    --save_dir "check/eval_results" \
    --debug_dir "$DEBUG_DIR"
