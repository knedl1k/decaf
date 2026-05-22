# MTG Card Recognition Pipeline

An end-to-end pipeline for training, evaluating, and deploying a deep learning model
(ResNet-50 + ArcFace) for Magic: The Gathering (MTG) card recognition. The
system supports distributed training on High-Performance Computing (HPC)
clusters and edge inference on hardware setups using a Raspberry Pi.

## Architecture & Workflow

The pipeline is built on a metric learning approach:

1. *Training:* A ResNet-50 backbone is trained with an ArcFace loss head on
reference images. The training process applies synthetic augmentations
(including perspective warps, color jitter, motion blur and simulated
lighting/shadows) to model real-world photo conditions.
2. *Indexing:* The trained backbone extracts normalized 512-dimensional
embeddings for all reference cards, saving them into a database file (`.pt` or `.npz`).
3. *Inference & Evaluation:* Real photos are automatically detected and
rectified into a top-down view via homography. The model extracts their
embeddings and searches for the closest match using cosine similarity. To handle
upside-down cards, the inference pipeline evaluates both the original
orientation and a 180-degree rotated crop, selecting the higher similarity score.

---

## Data Structure Requirements

The training and evaluation scripts expect specific naming conventions to
correctly parse the card name and expansion set edition from the filename. Any
trailing UUIDs or card numbers are automatically handled.

### Expected Directory Layout

```text
data/
├── reference_cards/
│   ├── aer_1_card-name.png       # Format: edition_collectorNumber_name
│   ├── khm_another-card.png      # Format: edition_name
│   └── m21_island-12345.png      # Format with a unique identifier trailing
└── real_photos/
    ├── photo_001.jpg             # Labeled real photos for validation/benchmarking
    └── photo_002.jpg
```

## Project Structure

- [**`data.py`**](./data.py): centralized dataset definitions, Albumentations augmentation
pipelines and image loading logic.
- [**`model.py`**](./model.py): neural network architecture comprising the
ResNet-50 backbone and ArcFace margin-based classification head.
- [**`utils.py`**](./utils.py): core utilities including homography-based smart cropping,
file parsing, and evaluation metrics.
- [**`lr_scheduler.py`**](./lr_scheduler.py): custom polynomial learning rate
scheduler with warmup heuristics.
- [**`train.py`**](./train.py): training orchestrator.
- [**`build_index.py`**](./build_index.py): feature extraction script for generating PyTorch
(`.pt`) and Edge/ONNX (`.npz`) vector databases.
- [**`evaluate_labeled.py`**](./evaluate_labeled.py): benchmarking script for
calculating Top-1/3/5 metrics on real-world test sets.
- [**`inference_rpi.py`**](./inference_rpi.py): lightweight, ONNX-accelerated inference script
optimized for edge devices (Raspberry Pi).

---

## Usage Guide

### Distributed Training (HPC / SLURM)

The training script utilizes PyTorch Distributed Data Parallel (DDP) for
multi-GPU configurations. Example command to launch on a single node with 2 GPUs
using torchrun:

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --ref_dir="$REF_DIR" \
    --real_val_dir="$REAL_DIR" \
    --save_dir="$SAVE_DIR" \
    --lr=0.1 \
    --epochs=60
```

### Generating the Feature Index

After training, generate the embedding database used for vector similarity matching
during inference.

```bash
srun python3 build_index.py \
    --model "$MODEL_PATH" \
    --images "$IMG_DIR" \
    --save_dir "$OUT_PATH"
```

### Evaluating on Labeled Real Photos

Run the benchmarking script over a directory of real photos to compute Top-1, Top-3,
Top-5 accuracy metrics:

```bash
python evaluate_labeled.py \
    --model ./checkpoints/arcface_mtg_ep25.pth \
    --database ./mtg_database.pt \
    --real_dir ./data/real_photos \
    --save_dir ./eval_results
```

### Exporting Model nad Database for Edge Deployment

To deploy the trained pipeline to the RPi, convert the model checkpoint to an
optimized ONNX graph and compress the database into a half-precision NumPy archive.

Script is prepared in [scripts](../../scripts) directory.

## Environment and Requirements

**Core Dependencies:**

- Python 3.13.5
- PyTorch 2.10.0 (CUDA 12.9.1)
- Albumentations 2.0.8
- OpenCV 4.12.0
- `timm` 1.0.25 (PyTorch Image Models)
- `matplotlib` and `numpy`
- `onnxruntime` (For Edge inference)

For environment initialization via Environment Modules (Lmod), execute:

```bash
module load Python/3.13.5-GCCcore-14.3.0
module load PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
module load Albumentations/2.0.8-foss-2025b-CUDA-12.9.1
module load OpenCV/4.12.0-foss-2025b-CUDA-12.9.1-contrib
module load timm/1.0.25-foss-2025b-CUDA-12.9.1
```
