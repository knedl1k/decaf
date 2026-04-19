# Training instructions

## 1. Environment and Requirements

The codebase is engineered to execute within a high-performance computing (HPC) environment utilizing the SLURM workload manager.

**Core Dependencies:**

- Python 3.13.5
- PyTorch 2.10.0 (CUDA 12.9.1)
- Albumentations 2.0.8
- OpenCV 4.12.0
- `timm` 1.0.25 (PyTorch Image Models)
- `matplotlib` and `numpy`

For environment initialization via Environment Modules (Lmod), execute:

```bash
module load Python/3.13.5-GCCcore-14.3.0
module load PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
module load Albumentations/2.0.8-foss-2025b-CUDA-12.9.1
module load OpenCV/4.12.0-foss-2025b-CUDA-12.9.1-contrib
module load timm/1.0.25-foss-2025b-CUDA-12.9.1
```

## 2. Dataset Preparation

The training process uses two directories:

1. Reference Directory (`--ref-dir`): Contains digital, canonical reference images of MTG cards. File names must act as the primary class labels (e.g., `6ed_273_ankh-of-mishra-c07903f1-defc-4ec7-accb-724b0219acd8.png`).
2. Real Validation Directory (`--real_val_dir`): (Optional but recommended) Contains real photographs of physical cards. Ground truth labels are dynamically parsed from the file stems using regex stripping.

## 3. Distributed Training Process

Training is orchestrated using PyTorch Distributed Data Parallel (DDP) across multiple GPUs. We utilize a ResNet-50 backbone pre-trained on ImageNet, fine-tuned using Stochastic Gradient Descent (SGD) with polynomial learning rate decay and a warmup heuristic.

### Launching the SLURM job

A predefined SLURM script (`job_train.sh`) is provided for automated cluster submission. It configures a single-node, 4-GPU topology.

### Technical Parameters within `job_train.sh`

The script utilizes torchrun to orchestrate the distributed backend. Key hyperparameters include:

- `--nnodes=1` and `--nproc_per_node=4`: Initializes 4 concurrent DDP worker processes.
- `--lr=0.1`: The base learning rate. Note that the codebase enforces a 10× learning rate multiplier discrepancy between the fully connected embedding layers and the convolutional backbone to preserve pre-trained spatial representations.
- `--epochs=60`: Total iterations over the reference dataset.

### Output and Artifacts

Upon execution, the training script yields the following artifacts within the specified `--save_dir`:

- `arcface_mtg_epX.pth`: State dictionaries comprising backbone weights, optimizer momentum buffers, and ArcFace class centers.
- `history.npy`: Serialized scalar logs for loss and evaluation metrics.
