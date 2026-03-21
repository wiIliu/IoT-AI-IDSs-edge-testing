# CSI4999 — Intrusion Detection at the Edge in IoT
![Python](https://img.shields.io/badge/python-3.12-blue) 
<a href="https://github.com/wiIliu/IoT-AI-IDSs-edge-testing/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-purple"></a>
[![Dataset: UCI RT-IoT2022](https://img.shields.io/badge/Dataset-UCI%20RT--IoT2022-blue)](https://archive.ics.uci.edu/dataset/942/rt-iot2022)
[![Dateset DOI](https://img.shields.io/badge/DOI-10.24432%2FC5P338-blue)](https://doi.org/10.24432/C5P338)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/legalcode)
[![Paper: Undergraduate Thesis](https://img.shields.io/badge/Paper-Undergraduate%20Thesis-orange)](./docs/willowConnelly-thesis-RealTimeIoTIDS-EvaluatingMLDLandEdgeDeploymentOnKriaKV260.pdf)


Investigating ML vs. DL-based intrusion detection systems (IDS) for IoT networks, with edge deployment on an FPGA.

A practical comparison demonstrating:

- **RT-IoT2022** usage for **binary (malicious or benign) and multi-class network traffic classification**
- **ML models** (Random Forest, XGBoost) for training and comparison
- **DL models** (CNN with and without self-attention) for training and comparison
- **Edge deployment** — baseline CNN quantized via Vitis AI and accelerated on a **Kria KV260 FPGA**

**Separate dataset download required** — https://doi.org/10.24432/C5P338

---

## Table of Contents

1. [Overview](#overview)
2. [Thesis](#thesis)
3. [Features](#features)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
   - [Creating & Activating a Conda Environment](#creating--activating-a-conda-environment)
   - [Installing Dependencies](#installing-dependencies)
6. [Usage](#usage)
   - [Data Preparation](#data-preparation)
   - [Model Training](#model-training)
   - [Evaluation](#evaluation)
7. [Results](#results)
   - [Binary Classification](#binary-classification)
   - [Multi-Class Classification](#multi-class-classification)
   - [Edge Deployment](#edge-deployment)
8. [Limitations & Future Work](#limitations--future-work)
9. [Contributing](#contributing)
10. [License](#license)
11. [References](#references)

---

## Overview

&nbsp;&nbsp;&nbsp;&nbsp;The Internet of Things (IoT) connects billions of task-specific devices across healthcare, industrial automation, smart homes, and more. This diversity makes it nearly impossible to define a single security solution, and the resource constraints of IoT hardware make traditional security approaches impractical. This project investigates the design of a real-time **intrusion detection system (IDS)** targeting **network-layer attacks** on IoT systems, and optimizes it for deployment at the network edge to act as a guardian of the IoT gateway.

&nbsp;&nbsp;&nbsp;&nbsp;Both ML and DL approaches are trained and benchmarked on the **RT-IoT2022** dataset for two tasks: **binary classification** (benign vs. malicious) and **multi-class classification** (12 specific traffic/attack types). The models evaluated are Random Forest (RF), XGBoost, a baseline CNN, and a CNN with self-attention. The most deployable model is then quantized via Vitis AI and accelerated on a **Kria KV260 FPGA**, demonstrating feasibility for real-time edge inference with minimal latency.

**Main contributions:**
- Training and evaluating RF, XGBoost, and CNN (± self-attention) on a recent IoT IDS dataset across both binary and multi-class objectives
- Quantitative comparison of ML vs. DL trade-offs in accuracy, F1, MCC, and computational overhead — relevant to resource-constrained IoT environments
- Proof-of-concept edge deployment of a quantized CNN on a Kria KV260 FPGA, demonstrating real-time inference capability at the network gateway

---

## Thesis

This repository originated as a group capstone project.  
The accompanying thesis was **independently authored** as part of the Oakland University Young's Honors College and expands on the ideas explored here.

- **Author:** Willow Connelly
- **Title:** _Real-Time Intrusion Detection for IoT Networks: Evaluating ML/DL Models and Feasibility of Edge Deployment on the Kria KV260_
- **Type:** Undergraduate Honors Thesis (Awarded: Thesis of Distinction)  
- **Scope:** Independent work. Contributions include data preprocessing, significant ML/DL modeling, and all edge deployment work.

[Read the thesis](./docs/willowConnelly-thesis-RealTimeIoTIDS-EvaluatingMLDLandEdgeDeploymentOnKriaKV260.pdf)

---

## Features

- **Binary Classification** — Classify network traffic as *Benign* or *Malicious*
- **Multi-Class Classification** — Identify 12 specific traffic/attack types (ARP poisoning, DDoS Slowloris, DOS SYN, MQTT, five NMAP scan variants, Metasploit SSH brute force, ThingSpeak, Wipro Bulb)
- **ML Baselines** — Random Forest and XGBoost with hyperparameter tuning
- **1D/2D CNN Models** — Convolutional networks treating features as 1D sequences or reshaped 2D grids
- **Self-Attention Variants** — CNN architectures with an added self-attention module; notably improves minority class discrimination (NMAP subtype separation)
- **Class Imbalance Handling** — SMOTE oversampling applied during training
- **Edge Deployment** — Baseline CNN quantized to a fixed-point `.xmodel` via Vitis AI and deployed on a Xilinx Kria KV260 DPU for accelerated real-time inference
- **Evaluation Suite** — Classification reports (accuracy, F1, MCC, ROC AUC), confusion matrices, and per-sample latency benchmarks saved per model

---

## Project Structure

```
.
├── dataPreprocessScript.py               # Loads and splits RT-IoT2022 (80/20 train/test)
├── dataAnalysisVisualization.ipynb       # Dataset analysis and class distribution visualization
│
├── ml_binary_trainingValidation.ipynb    # ML binary: XGBoost, RF — trains, tunes, validates, exports
├── ml_multi_trainingValidation.ipynb     # ML multi-class: XGBoost, RF, CatBoost — trains, tunes, validates, exports
│
├── 1dcnn_binary.ipynb                    # 1D CNN binary classification
├── 1dcnn_binary selfattn.ipynb           # 1D CNN + self-attention binary classification
├── 1dcnn_multi.ipynb                     # 1D CNN multi-class classification
├── 1dcnn_multi_selfattn.ipynb            # 1D CNN + self-attention multi-class classification
│
├── latencyTiming.ipynb                   # Per-sample latency benchmarking across models and hardware
│
├── models/                               # Exported model checkpoints
│   ├── 1dcnn_binary.pth
│   ├── 1dcnn_binary_attn.pth
│   ├── 1dcnn_multiclass.pth
│   ├── 1dcnn_multiclass_attn.pth
│   ├── best_binary_model.pkl             # Best ML binary model
│   ├── best_multi_ml.pkl                 # Best ML multi-class model
│   ├── rf_binary_model.pkl / rf_multi_model.pkl
│   ├── xgb_binary_model.pkl / xgb_multi_model.pkl
│   └── catboost_binary_model.pkl / catboost_multi_model.pkl
│
├── classification_reports/               # CSV evaluation metrics per model
├── confusionmatrices/                    # Confusion matrix PNGs per model
│
├── edge/                                 # Edge deployment (Vitis AI / Kria KV260)
│   ├── 2dcnn_binary.ipynb                # 2D CNN binary classification
│   ├── 2dcnn_binary selfattn.ipynb       # 2D CNN + self-attention binary
│   ├── 2dcnn_multi.ipynb                 # 2D CNN multi-class classification
│   ├── 2dcnn_multi_selfattn.ipynb        # 2D CNN + self-attention multi-class
│   ├── BinaryCNN_classFile.py            # Model class definition for binary 2D CNN
│   ├── MultiCNN_classFile.py             # Model class definition for multi-class 2D CNN
│   ├── MultiAttnCNN_classFile.py         # Model class definition for multi-class 2D CNN + attention
│   ├── binaryQuantizer.py                # Quantizes binary CNN model (Vitis AI)
│   ├── multiQuantizer.py                 # Quantizes multi-class CNN model (Vitis AI)
│   ├── binaryTestLoader.py               # DataLoader for binary calibration/eval
│   ├── multiTestLoader.py                # DataLoader for multi-class calibration/eval
│   ├── edgeDataPreparer.py               # Prepares .npy data arrays for edge use
│   ├── inference.py                      # On-device inference script
│   ├── inference-multi.ipynb             # Multi-class inference notebook
│   ├── cap_binaryCNN_first.xmodel        # Quantized binary CNN (Vitis AI xmodel)
│   ├── cap_multiCNN_first.xmodel         # Quantized multi-class CNN (Vitis AI xmodel)
│   └── cnn_compile.sh                    # Vitis AI compile script
│
├── webdemo/                              # Project website
│   ├── index.html
│   ├── methodology.html
│   ├── challenges.html
│   ├── results.html
│   ├── future-work.html
│   ├── server.js
│   └── static/
│       ├── app.js
│       └── style.css
│
├── environment.yml                       # Conda environment definition
└── requirements.txt                      # pip dependencies
```

---

## Installation

### Creating & Activating a Conda Environment

1. **Install** [Anaconda/Miniconda](https://docs.conda.io/en/latest/) if you haven't already.
2. **Create** a new environment:
   ```bash
   conda create -n iot_ids python=3.12
   ```
3. **Activate** it:
   ```bash
   conda activate iot_ids
   ```

### Installing Dependencies

Within your **activated** conda environment:

```bash
conda env create -f environment.yml
```

Or, if you prefer `pip`:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- Python 3.12, PyTorch 2.x
- scikit-learn, imbalanced-learn (SMOTE)
- XGBoost, CatBoost
- Pandas, Matplotlib

---

## Usage

![projectProcessSmote](https://github.com/user-attachments/assets/e72ecc4c-8730-4c8f-9c72-8ed198ae391f)

### 1. Data Preparation

**Manual download required** from the UCI ML Repository: https://doi.org/10.24432/C5P338

After downloading, run `dataPreprocessScript.py` to split the dataset into **train** and **test** sets (80/20 split).

```bash
python dataPreprocessScript.py
```

Optionally open `dataAnalysisVisualization.ipynb` to explore class distributions and feature statistics before training.

### 2. Model Training

Each notebook is self-contained — open and run top-to-bottom in Jupyter:

| Task | Notebook |
|------|----------|
| ML binary (XGBoost, RF) | `ml_binary_trainingValidation.ipynb` |
| ML multi-class (XGBoost, RF, CatBoost) | `ml_multi_trainingValidation.ipynb` |
| 1D CNN binary | `1dcnn_binary.ipynb` |
| 1D CNN binary + self-attention | `1dcnn_binary selfattn.ipynb` |
| 1D CNN multi-class | `1dcnn_multi.ipynb` |
| 1D CNN multi-class + self-attention | `1dcnn_multi_selfattn.ipynb` |
| 2D CNN binary (edge-targeted) | `edge/2dcnn_binary.ipynb` |
| 2D CNN binary + self-attention | `edge/2dcnn_binary selfattn.ipynb` |
| 2D CNN multi-class (edge-targeted) | `edge/2dcnn_multi.ipynb` |
| 2D CNN multi-class + self-attention | `edge/2dcnn_multi_selfattn.ipynb` |

Trained model files are saved to `models/` (`.pth` for PyTorch, `.pkl` for scikit-learn/XGBoost).

### 3. Evaluation

Classification reports and confusion matrices are automatically saved to `classification_reports/` and `confusionmatrices/` when running each training notebook. For latency benchmarking across hardware, see `latencyTiming.ipynb`.

For edge inference, see `edge/inference.py` and `edge/inference-multi.ipynb`.

---

## Results

### Binary Classification

| Model | Accuracy | Precision | Recall | F1 | MCC |
|-------|----------|-----------|--------|----|-----|
| Random Forest | 0.9984 | 0.9988 | 0.9994 | **0.9991** | 0.9913 |
| XGBoost | 0.9983 | 0.9994 | 0.9987 | **0.9991** | 0.9909 |
| CNN (baseline) | 0.9913 | 0.9940 | 0.9964 | 0.9952 | 0.9522 |
| CNN + Self-Attention | 0.9900 | 0.9911 | 0.9978 | 0.9945 | 0.9443 |

ML ensemble models achieved the strongest binary performance. The baseline CNN, while slightly behind, reached a competitive F1 of 99.52% and — critically — is the only architecture compatible with full quantization and DPU deployment on the target hardware.

### Multi-Class Classification (12 classes)

| Model | Accuracy | Weighted F1 | Macro F1 | MCC |
|-------|----------|-------------|----------|-----|
| XGBoost | 0.9981 | 0.9980 | **0.9792** | 0.9951 |
| Random Forest | 0.9980 | 0.9980 | 0.9569 | 0.9949 |
| CNN (baseline) | 0.9493 | 0.9406 | 0.5324 | 0.8753 |
| CNN + Self-Attention | 0.9905 | **0.9912** | 0.8686 | 0.9762 |

The baseline CNN struggled with the five NMAP scan subclasses, collapsing them into a single predicted class. The self-attention variant resolved this by learning to weight the discriminative features between scan types, improving macro F1 by over 30 percentage points. XGBoost achieved the best overall multi-class performance.

### Edge Deployment

Only the **baseline CNN** was quantized and deployed to the Kria KV260 DPU (the attention variant was not compatible with the target DPU configuration).

**Binary — per-sample inference latency:**

| Model | Hardware | Latency |
|-------|----------|---------|
| CNN (baseline) | Kria KV260 DPU | **0.18 ms** |
| CNN (baseline) | Intel i7 CPU | ~0.45 ms |
| XGBoost | Intel i7 CPU | — |
| Random Forest | Intel i7 CPU | 10.73 ms (59× slower than DPU) |

**Multi-class — per-sample inference latency:**

| Model | Hardware | Latency |
|-------|----------|---------|
| CNN (baseline) | Kria KV260 DPU | **0.21 ms** |
| CNN (baseline) | Intel i7 CPU | 0.71 ms (3.25× slower) |
| XGBoost | Intel i7 CPU | ~7.4 ms (35× slower than DPU) |

**Post-quantization performance:**
- *Binary*: F1 decreased marginally to 99.14%, MCC dropped 3.27 points to 91.95% — trade-off is minimal
- *Multi-class*: Macro F1 **increased** from 53.24% to 83.23% after quantization — suggesting the original model was overfit and quantization acted as a regularizer

> The successful deployment demonstrates that edge-accelerated CNNs can provide near real-time threat detection at the IoT gateway with latency well under 1 ms per sample, while ML models — despite higher classification accuracy — are orders of magnitude slower and not quantization-compatible with the target hardware.

---

## Limitations & Future Work

**Limitations:**
- The RT-IoT2022 dataset lacks temporal/sequential information, which limits the theoretical benefit of convolutional operators (features have no inherent spatial ordering)
- Class imbalance remains a challenge for rare attack types even with SMOTE; the NMAP subclasses have very few samples
- Only baseline CNN architectures were tested on the DPU due to time constraints — attention variants and other architectures were not evaluated on hardware

**Future directions:**
- Evaluate on more realistic, temporally-rich IoT network datasets to better leverage CNN sequence-learning capability
- Assess quantization-aware training (QAT) to reduce the performance gap introduced by post-training quantization
- Compare alternative edge hardware platforms beyond the Kria KV260
- Design a lightweight data ingestion and transformation pipeline suitable for live edge environments
- Explore hybrid DL architectures (e.g., CNN + LSTM, transformer-based) for improved minority class detection

---

## Contributing

1. **Fork** this repository.
2. **Create** a new branch for your feature/fix:
   ```bash
   git checkout -b feature-my-improvement
   ```
3. **Commit** your changes and push to your fork:
   ```bash
   git commit -m "Add my new feature"
   git push origin feature-my-improvement
   ```
4. **Open a Pull Request** into the main branch.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## References

1. **RT-IoT2022 Dataset** — B. S. and R. Nagapadma, "RT-IoT2022," UCI Machine Learning Repository, 2023. DOI: https://doi.org/10.24432/C5P338.

---

_Thank you for visiting **IoT-AI-IDSs-edge-testing**! If you have any questions or issues, feel free to open an [issue](../../issues) or reach out._
