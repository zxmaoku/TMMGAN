# TMM-GAN: Trusted Multi-Modal Generative Adversarial Networks for Missing Image Synthesis

This repository contains the official PyTorch implementation of TMM-GAN, as described in:

> Shenglei Pei, Shoupeng Zhang, Xin Zhang, and Zepu Hao. "Trusted Multi-Modal Generative Adversarial Networks for Missing Image Synthesis." *Frontiers in Artificial Intelligence*, 2026.

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Environment Setup](#3-environment-setup)
- [4. Dataset Preparation](#4-dataset-preparation)
- [5. Training Details](#5-training-details)
- [6. Evaluation Metrics](#6-evaluation-metrics)
- [7. Citation](#7-citation)

---

## 1. Overview

TMM-GAN is a framework for synthesizing missing image modalities from complete multimodal inputs. It integrates adversarial training with autoencoder-guided representation learning. The framework consists of two core components:

### 1.1 Multimodal Translation Network

The multimodal translation network extracts hierarchical complementary cross-modal features from complete multimodal inputs and fuses them to generate the target (missing) modality image. It comprises:

- **Dual-layer Encoder**: A two-layer encoder structure that differentially captures hierarchical visual representations. The first encoder (Level 1) extracts low-level features, and the second encoder (Level 2) extracts high-level features. Each encoder uses **DEConv** (Detail-Enhancing Convolution) and **DEB/DEAB blocks** (detail-enhanced attention blocks) from DEA-NET (Chen et al., 2024).

- **CGA-Fusion**: Content-Guided Attention Fusion modules that integrate multi-scale representations from different modalities via spatial attention, channel attention, and pixel attention.

- **Parallel Autoencoder**: An autoencoder network (with the same DEA-NET structure) operating in parallel under target modality constraints. It extracts high/low-level features from the target modality to guide the encoder and fusion modules of the multimodal translation network via feature loss.

### 1.2 Trusted Discriminator Network (TMC)

A Trusted Multi-view Classifier (TMC) serves as the discriminator, replacing traditional adversarial training with reliability calibration constraints. Based on ResNet18, it models epistemic uncertainty using **Dirichlet distribution** evidence theory (Dempster-Shafer theory). The TMC not only distinguishes real from synthetic images but also extracts supporting evidence for classification, providing deeper supervision on the generator.

## 2. Repository Structure

```
TMMGAN-main/
├── README.md
├── dataset/                          # Place BraTS2020 data here
└── TMMgan/
    ├── DEANET.py                     # DEANet encoder & DEA_Decoder (autoencoder structure)
    ├── tmc.py                        # Trusted Multi-view Classifier (TMC discriminator)
    ├── index.py                      # Evaluation metrics (PSNR, SSIM, LPIPS)
    ├── createmiss.py                 # Missing-modality data augmentation utility
    ├── data.py                       # Dataset utilities
    ├── dataL.py                      # Data loading utilities
    ├── T1_train.py                   # Training/testing script for T1 modality
    ├── T1gd_train.py                 # Training/testing script for T1Gd modality
    ├── T2_train.py                   # Training/testing script for T2 modality
    ├── flair_train.py                # Training/testing script for FLAIR modality
    ├── train.py                      # Base training utilities
    ├── newtrain.py                   # Alternative training entry point
    ├── cgtest.py                     # Testing utilities
    ├── nilltrans.py                  # Additional transforms
    └── modules/
        ├── __init__.py
        ├── deconv.py                 # Detail-Enhancing Convolution (DEConv)
        ├── deablock.py               # DEA blocks (inference)
        ├── deablock_train.py         # DEA blocks (training)
        ├── cga.py                    # Spatial/Channel/Pixel attention
        ├── fusion.py                 # CGA-Fusion module
        └── image.py                  # Image encoder utilities
```

---

## 3. Environment Setup

### 3.1 System Requirements

- **OS**: Linux / Windows
- **Python**: 3.8+
- **GPU**: NVIDIA GPU (experiments in the paper were conducted on a single NVIDIA A800 GPU)
- **PyTorch**: 2.1.2+cu118

### 3.2 Create Environment

```bash
conda create -n tmmgan python=3.8
conda activate tmmgan
```

### 3.3 Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 3.4 Install Dependencies

Before running the code, set up the project environment with the following packages:

```bash
pip install numpy==1.26.4 scipy==1.15.3 pandas==2.2.3
pip install einops==0.8.1 tqdm==4.65.2
pip install opencv-python==4.11.0.86 pillow==11.0.0
pip install matplotlib==3.10.3 seaborn==0.13.2
pip install scikit-learn==1.7.0
pip install kornia lpips
pip install transformers==4.52.4 timm==1.0.15
```

> **Note**: `kornia` and `lpips` are required for computing the PSNR, SSIM, and LPIPS evaluation metrics (see [index.py](TMMgan/index.py)).

### 3.5 Full Dependency List

The complete pinned environment (for exact reproducibility) is listed below:

<details>
<summary>Click to expand the full package list</summary>

```
accelerate             1.8.0
addict                 2.4.0
aiohappyeyeballs       2.6.1
aiohttp                3.12.13
aiosignal              1.3.2
async-timeout          5.0.1
attempt                0.1.1
attrs                  25.3.0
beautifulsoup4         4.13.4
bs4                    0.0.2
certifi                2025.6.15
cffi                   2.0.0
charset-normalizer     2.1.1
click                  8.3.0
colorama               0.4.6
contourpy              1.3.2
cryptography           46.0.3
cycler                 0.12.1
datasets               3.6.0
dill                   0.3.8
einops                 0.8.1
exceptiongroup         1.3.0
faiss-cpu              1.11.0
filelock               3.14.0
fonttools              4.58.1
frozenlist             1.7.0
fsspec                 2024.6.1
h11                    0.16.0
huggingface-hub        0.32.2
idna                   3.4
iniconfig              2.1.0
jieba                  0.42.1
Jinja2                 3.1.4
jmespath               0.10.0
joblib                 1.5.1
kiwisolver             1.4.8
labelImg               1.8.6
lxml                   6.0.1
Markdown               3.9
markdown-it-py         4.0.0
MarkupSafe             2.1.5
matplotlib             3.10.3
mdurl                  0.1.2
model-index            0.1.11
mpmath                 1.3.0
multidict              6.5.0
multiprocess           0.70.16
networkx               3.3
numpy                  1.26.4
opencv-python          4.11.0.86
ordered-set            4.1.0
outcome                1.3.0.post0
packaging              24.2
pandas                 2.2.3
pillow                 11.0.0
pip                    25.1
platformdirs           4.5.0
pluggy                 1.6.0
polars                 1.33.1
propcache              0.3.2
psutil                 7.0.0
py-cpuinfo             9.0.0
pyarrow                20.0.0
pycparser              2.22
pycryptodome           3.23.0
Pygments               2.19.2
pyparsing              3.2.3
PyQt5                  5.15.11
PyQt5-Qt5              5.15.2
PyQt5_sip              12.17.0
PySocks                1.7.1
pytest                 8.4.2
python-dateutil        2.9.0.post0
pytz                   2023.4
pywin32                311
PyYAML                 6.0.2
regex                  2024.11.6
requests               2.28.2
rich                   13.4.2
safetensors            0.5.3
scikit-learn           1.7.0
scipy                  1.15.3
seaborn                0.13.2
sentence-transformers  4.1.0
sentencepiece          0.2.0
setuptools             60.2.0
six                    1.17.0
sniffio                1.3.1
sortedcontainers       2.4.0
soupsieve              2.7
sympy                  1.13.3
tabulate               0.9.0
thop                   0.1.1.post2209072238
threadpoolctl          3.6.0
timm                   1.0.15
tokenizers             0.21.1
tomli                  2.3.0
torch                  2.1.2+cu118
torchaudio             2.1.2+cu118
torchvision            0.16.2+cu118
tqdm                   4.65.2
transformers           4.52.4
trio                   0.30.0
trio-websocket         0.12.2
typing_extensions      4.13.2
tzdata                 2025.2
ultralytics            8.3.146
ultralytics-thop       2.0.14
urllib3                1.26.20
websocket-client       1.8.0
wheel                  0.45.1
wsproto                1.2.0
xxhash                 3.5.0
yapf                   0.43.0
yarl                   1.20.1
```

</details>

---

## 4. Dataset Preparation

### 4.1 Dataset Download

The processed dataset used in this work can be downloaded from the `dataset/` folder in this repository ([TMMGAN-main/dataset](dataset)).

### 4.2 Organize the Data

Extract axial 2D slices from the 3D NIfTI volumes and save them as PNG images (grayscale). Organize the data into the following directory structure:

```
dataset/
├── BraTS2020_TrainingData/
│   ├── t1/          # T1 modality slices (.png)
│   ├── t1gd/        # T1Gd modality slices (.png)
│   ├── t2/          # T2 modality slices (.png)
│   └── flair/       # FLAIR modality slices (.png)
└── BraTS2020_ValidationData/
    ├── t1/
    ├── t1gd/
    ├── t2/
    └── flair/
```

Each subfolder should contain PNG images of the same patient/slice across all four modalities (aligned by filename).

### 4.3 Configure Data Paths

The data paths are defined in each training script. Open the relevant training file and modify the `train_paths` and `test_paths` variables to point to your local dataset:

```python
# Example in T1_train.py (lines 82-87)
train_paths = [
    '/path/to/BraTS2020_TrainingData/t1',
    '/path/to/BraTS2020_TrainingData/t1gd',
    '/path/to/BraTS2020_TrainingData/t2',
    '/path/to/BraTS2020_TrainingData/flair'
]
test_paths = [
    '/path/to/BraTS2020_ValidationData/t1',
    '/path/to/BraTS2020_ValidationData/t1gd',
    '/path/to/BraTS2020_ValidationData/t2',
    '/path/to/BraTS2020_ValidationData/flair'
]
```

> **Important**: Update these paths in all four training scripts (`T1_train.py`, `T1gd_train.py`, `T2_train.py`, `flair_train.py`) before running.

---

## 5. Training Details

### 5.1 Experimental Setup (from the paper)

- **Framework**: PyTorch
- **GPU**: Single NVIDIA A800 GPU
- **Batch size**: 32
- **Image size**: 256×256


## 6. Evaluation Metrics

The model is evaluated using three standard image quality metrics (computed in [index.py](TMMgan/index.py)). For SSIM and PSNR, higher scores indicate better performance; for LPIPS, lower is better. All reported evaluation scores are computed between the synthetic images and their corresponding ground truth.

| Metric | Library | Description |
|--------|---------|-------------|
| **SSIM** | kornia | Structural Similarity Index Measure — evaluates image-level consistency (↑ higher is better) |
| **PSNR** | kornia | Peak Signal-to-Noise Ratio — measures reconstruction quality (↑ higher is better) |
| **LPIPS** | lpips (AlexNet) | Learned Perceptual Image Patch Similarity — measures perceptual distance (↓ lower is better) |

---

## 7. Citation

If you find this work useful, please cite our paper:

```bibtex
@article{pei2026tmmgan,
  title={Trusted Multi-Modal Generative Adversarial Networks for Missing Image Synthesis},
  author={Pei, Shenglei and Zhang, Shoupeng and Zhang, Xin and Hao, Zepu},
  journal={Frontiers in Artificial Intelligence},
  year={2026}
}
```
