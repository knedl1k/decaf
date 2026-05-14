# MTG Card Recognition Pipeline

An end-to-end pipeline for training, evaluating, and deploying a deep learning
model (ResNet-50 + ArcFace) for Magic: The Gathering (MTG) card recognition.

## Project Structure

- **`data.py`**: centralized dataset definitions, Albumentations augmentation
pipelines and image loading logic.
- **`model.py`**: neural network architecture comprising the ResNet-50 backbone
and ArcFace margin-based classification head.
- **`utils.py`**: core utilities including homography-based smart cropping,
file parsing, and evaluation metrics.
- **`lr_scheduler.py`**: custom polynomial learning rate scheduler with warmup heuristics.
- **`train.py`**: training orchestrator.
- **`create_database.py`**: Feature extraction script for generating PyTorch
(`.pt`) and Edge/ONNX (`.npz`) vector databases.
- **`evaluate.py`**: benchmarking script for calculating Top-1/3/5 metrics on
real-world test sets.
- **`inference_rpi.py`**: Lightweight, ONNX-accelerated inference script
optimized for edge devices (Raspberry Pi).

---

## 1. Environment and Requirements

The codebase is prepared to be executed within a high-performance computing (HPC)
environment utilizing the SLURM workload manager.

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
