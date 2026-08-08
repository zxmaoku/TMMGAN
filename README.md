# TMM-GAN: Trusted Multi-Modal Generative Adversarial Networks for Missing Image Synthesis

This repository contains the official PyTorch implementation of TMM-GAN, as described in:

> Shenglei Pei, Shoupeng Zhang, Xin Zhang, and Zepu Hao. "Trusted Multi-Modal Generative Adversarial Networks for Missing Image Synthesis." *Frontiers in Artificial Intelligence*, 2026.

If you encounter any problems with code execution or dataset downloads, please contact us by email: **zxsurefire@163.com**

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Environment Setup](#3-environment-setup)
- [4. Dataset Preparation](#4-dataset-preparation)
- [5. Reproducing the Numerical Results](#5-reproducing-the-numerical-results)
- [6. Training Details](#6-training-details)
- [7. Evaluation Metrics](#7-evaluation-metrics)
- [8. Expected Results](#8-expected-results)
- [9. Citation](#9-citation)

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

### 1.3 Loss Functions

The total generator loss combines five terms:

```
L_G = L_mk + L_Re + L_f + L_adv + L_E
```

| Loss | Description |
|------|-------------|
| `L_AE` | Autoencoder reconstruction loss (L1) between generated and input images |
| `L_Re` | Reconstruction loss for common (available) modalities |
| `L_mk` | Target modality reconstruction loss (L1) |
| `L_f` | Feature loss guiding encoder branches via autoencoder deep supervision |
| `L_adv` | Adversarial loss from the trusted discriminator |
| `L_E` | Evidence loss (KL divergence between fake and real evidence distributions) |

---

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

### 4.1 BraTS2020 Database (Medical Image Synthesis)

The model is trained and evaluated on the [BraTS2020 (Brain Tumor Segmentation 2020)](https://www.med.upenn.edu/cbica/brats2020/) database. The database includes four MRI imaging modalities:

- **T1**: native T1-weighted
- **T1Gd**: contrast-enhanced T1-weighted
- **T2**: T2-weighted
- **FLAIR**: T2 fluid-attenuated inversion recovery

A total of **494 patients** with high-grade or low-grade gliomas are included. The data are aligned to the same anatomical template, interpolated to the same resolution (1×1×1 mm³), and cropped. Each MRI image has a size of 155×240×240. In the official partition protocol, the training set contains images from **369 subjects**, and the remaining **125 subjects** are used as the test set.

Since the 3D MR images are scanned in the axial plane, 2D slices are extracted from each sample according to the scanning direction and resized to **256×256**.

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

## 5. Reproducing the Numerical Results

### 5.1 Quick Start

Each target modality has a dedicated training script. To reproduce the results for all four target modalities, run each script:

```bash
cd TMMgan

# Synthesize T1 modality (using T1Gd, T2, FLAIR as inputs)
python T1_train.py

# Synthesize T1Gd modality (using T1, T2, FLAIR as inputs)
python T1gd_train.py

# Synthesize T2 modality (using T1, T1Gd, FLAIR as inputs)
python T2_train.py

# Synthesize FLAIR modality (using T1, T1Gd, T2 as inputs)
python flair_train.py
```

Each script automatically:
1. Trains the model (calling `train()`)
2. Evaluates on the test set (calling `test()`)
3. Saves quantitative metrics (PSNR, SSIM, LPIPS) to CSV files

### 5.2 Output Files

After training and testing, the following files are generated in the `TMMgan/` directory:

**Model checkpoints** (saved every 50 epochs):
| Target Modality | Checkpoint Files |
|----------------|------------------|
| T1 | `T1_DEA_AE.pt`, `T1_DEA_DE.pt`, `T1_l1_mix.pt`, `T1_l2_mix.pt`, `T1_TMC.pt` |
| T1Gd | `T1gd_DEA_AE.pt`, `T1gd_DEA_DE.pt`, `T1gd_l1_mix.pt`, `T1gd_l2_mix.pt`, `T1gd_TMC.pt` |
| T2 | `T2_DEA_AE.pt`, `T2_DEA_DE.pt`, `T2_l1_mix.pt`, `T2_l2_mix.pt`, `T2_TMC.pt` |
| FLAIR | `flair_DEA_AE_1_1_0.3.pt`, `flair_DEA_DE_1_1_0.3.pt`, `flair_l1_mix_1_1_0.3.pt`, `flair_l2_mix_1_1_0.3.pt`, `flair_TMC.pt` |

**Metrics CSV files** (saved every 5 epochs):
| Target Modality | Train Score File | Test Score File |
|----------------|------------------|-----------------|
| T1 | `T1_score_list_train.csv` | `T1_score_list_test.csv` |
| T1Gd | `T1gd_score_list_train.csv` | `T1gd_score_list_est.csv` |
| T2 | `T2_score_list_train.csv` | `T2_score_list_test.csv` |
| FLAIR | `flair_score_list_train.csv` | `flair_score_list_test.csv` |

Each CSV contains three columns: `psnr`, `ssim`, `lpips`.

### 5.3 Resume Training

The training scripts support resuming from checkpoints. To resume training, ensure the corresponding `.pt` checkpoint files exist in the `TMMgan/` directory. The scripts automatically load them at the start of `train()` and `test()` functions.

---

## 6. Training Details

### 6.1 Experimental Setup (from the paper)

- **Framework**: PyTorch
- **GPU**: Single NVIDIA A800 GPU
- **Batch size**: 32
- **Input/output channels**: 1 (for medical image synthesis)
- **Image size**: 256×256
- **Pixel normalization**: Pixel values normalized to the range of [-1, 1] (equivalent to `mean=0.5, std=0.5`)

### 6.2 Code Hyperparameters

The following hyperparameters are defined in the training scripts (see each `*_train.py` file):

| Parameter | T1 | T1Gd | T2 | FLAIR |
|-----------|-----|------|-----|-------|
| Batch size | 32 | 32 | 32 | 32 |
| Training epochs | 400 | 150 | 500 | 500 |
| Testing epochs | 400 | 500 | 500 | 500 |
| Generator LR (`G_lr`) | 2e-6 | 1e-6 | 1e-6 | 2e-6 |
| TMC LR (`TMC_lr`) | 1e-6 | 1e-5 | 1e-5 | 1e-5 |
| Optimizer | Adamax | Adamax | Adamax | Adamax |
| `D_alpha` (adversarial) | 1 | 1 | 1 | 1 |
| `com_mse_alpha` | 0.5 | 1 | 1.5 | 1.5 |
| `target_mse_alpha` | 2 | 2 | 2 | 2 |
| `f_alpha` (feature loss) | 0.3 | 0.3 | 0.5 | 0.5 |
| `E_alpha` (evidence loss) | 0.1 | 0.1 | 0.1 | 0.1 |

### 6.3 Data Preprocessing

Input images are preprocessed as follows (consistent with the paper's normalization to [-1, 1]):

```python
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Maps [0,1] to [-1,1]
])
```

---

## 7. Evaluation Metrics

The model is evaluated using three standard image quality metrics (computed in [index.py](TMMgan/index.py)). For SSIM and PSNR, higher scores indicate better performance; for LPIPS, lower is better. All reported evaluation scores are computed between the synthetic images and their corresponding ground truth.

| Metric | Library | Description |
|--------|---------|-------------|
| **SSIM** | kornia | Structural Similarity Index Measure — evaluates image-level consistency (↑ higher is better) |
| **PSNR** | kornia | Peak Signal-to-Noise Ratio — measures reconstruction quality (↑ higher is better) |
| **LPIPS** | lpips (AlexNet) | Learned Perceptual Image Patch Similarity — measures perceptual distance (↓ lower is better) |

---

## 8. Expected Results

### 8.1 Medical Image Synthesis (BraTS2020)

The following table reports the quantitative comparison on the BraTS2020 database (Table 1 in the paper). The proposed TMM-GAN achieves the highest SSIM and PSNR scores, as well as the optimal (smallest) LPIPS values in most modalities.

| Methods | T1 (SSIM↑/PSNR↑/LPIPS↓) | T1Gd (SSIM↑/PSNR↑/LPIPS↓) | T2 (SSIM↑/PSNR↑/LPIPS↓) | FLAIR (SSIM↑/PSNR↑/LPIPS↓) |
|---------|------|------|------|------|
| CollaGAN | 0.7231/16.94/0.2919 | 0.7891/22.26/0.2101 | 0.7887/21.43/0.2272 | 0.8091/22.88/0.1892 |
| TSIT | 0.8239/22.42/0.1312 | 0.8634/26.25/0.1330 | 0.8335/22.46/0.1625 | 0.8392/22.25/0.1735 |
| CUT | 0.8555/23.18/0.1302 | 0.8733/26.21/0.1467 | 0.8172/21.18/0.1334 | 0.8134/20.52/0.1591 |
| AttentionGAN | 0.8587/23.58/0.1266 | 0.8732/26.56/0.1215 | 0.8568/23.37/0.1086 | 0.8231/21.23/0.1322 |
| TokenFusion | 0.8566/14.52/0.1166 | 0.8482/15.95/0.1503 | 0.9011/25.67/0.0932 | 0.8435/17.87/0.1396 |
| HMS-MambaGAN | 0.8987/26.70/0.0878 | 0.9234/29.28/0.0975 | 0.9178/28.45/0.0914 | 0.8962/21.75/0.1199 |
| FDCGAN | 0.9014/28.33/0.0972 | 0.9158/28.35/0.1011 | 0.9214/28.97/0.0895 | 0.8971/21.76/0.1185 |
| CSPMotifsGAN | 0.9011/28.41/0.1025 | 0.9142/28.61/0.1024 | 0.9207/28.73/0.0922 | 0.8891/21.83/0.1215 |
| **Ours (TMM-GAN)** | **0.9085/28.52/0.0825** | **0.9229/29.26/0.0924** | **0.9225/28.92/0.0872** | **0.8983/22.14/0.1160** |

### 8.2 Ablation Study (BraTS2020)

The ablation study validates the effectiveness of feature loss (`L_f`) and evidence loss (`L_E`) (Table 3 in the paper):

| Setting | SSIM↑ | PSNR↑ | LPIPS↓ |
|---------|-------|-------|--------|
| TMM-GAN (full) | 0.9139 | 27.21 | 0.0965 |
| w/o `L_f` | 0.9105 | 26.62 | 0.1023 |
| w/o `L_E` | 0.9113 | 26.81 | 0.1011 |
| w/o `L_f` and `L_E` | 0.9064 | 26.53 | 0.1022 |

### 8.3 Reading the Results from CSV Files

After training and testing, the final metrics can be read from the `*_score_list_test.csv` files (last row):

```python
import pandas as pd
df = pd.read_csv('T1_score_list_test.csv')
print(df.iloc[-1])  # Final epoch metrics: psnr, ssim, lpips
```

---

## 9. Citation

If you find this work useful, please cite our paper:

```bibtex
@article{pei2026tmmgan,
  title={Trusted Multi-Modal Generative Adversarial Networks for Missing Image Synthesis},
  author={Pei, Shenglei and Zhang, Shoupeng and Zhang, Xin and Hao, Zepu},
  journal={Frontiers in Artificial Intelligence},
  year={2026}
}
```

---

## Contact

For questions regarding code execution or dataset setup, please contact: **zxsurefire@163.com**

## Funding

This work was supported by the National Natural Science Foundation of China under Grant Nos. 62266035.
