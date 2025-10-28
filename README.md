# CSI4999 IDS in IoT using AI Solutions
Is there significant difference between ML vs DL-based models when used as an intrusion detection system in IoT environments. <br><br>

A practical comparision demonstrating:

- **RT-IoT-2022** usage for **binary (malicious or benign) and multi-class network traffic classification**  
- **ML models** (ensemble-based) for training and comparison
- **DL model** (CNN-based) for training and comparison
- **Inference and Testing**

**Separate dataset download required** - https://doi.org/10.24432/C5P338 

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
   - [Creating & Activating a Conda Environment](#creating--activating-a-conda-environment)  
   - [Installing Dependencies](#installing-dependencies)  
5. [Usage](#usage)  
   - [Data Preparation](#data-preparation)  
   - [Model Training](#model-training)  
   - [Evaluation](#evaluation)  
6. [Results](#results)  
7. [Contributing](#contributing)  
8. [License](#license)  
9. [References](#references)

---

## Overview

This repository, **CSI4999_IDSwithAI**, demonstrates ML and DL internet network classification on **RT-IoT2022**. After training, the project saves a checkpoint (`{model}.pth`) that you can copy for usage in desired ways to perform inference.<br>
This project focuses on building AI models to classify Internet of things (IoT) network data in two ways: as either benign or malicious, and multi-class. We  implement and compare two approaches: traditional machine learning (ML) methods using models, such as XGBoost and decision trees, and a deep learning (DL) method using a CNN model that is able to identify complex, non-linear relationships. <br>
By evaluating both approaches, we aim to identify which model performs better as an intrusion detection system (IDS)  in terms of accuracy, precision, recall, and overall robustness.<br>
In addition to the modeling work, we will develop a project website to clearly present our goals, methodology, results, challenges, and potential future work. This website will serve as a professional showcase of our project outcomes and will make our research accessible to both technical and non-technical audiences.

---

## Features


## Project Structure

You have **1** main Python files:

```
.
├──── models/ # holds all .pth model exports
├──── webdemo/ # code for website demo and info
| ├── index.html
| ├── methodology.html
| ├── challenges.html
| ├── results.html
| ├── future-work.html
| ├── server.js
| ├──── static/
| | ├── app.js
| | ├── style.css
├── dataPreprocessing.ipynb   # Loads and splits RT-IoT2022; Visualizes data structure
├── ml_trainingValidation.ipynb   # Builds, tunes, trains, validates, and exports ML models for binary classification
├── ml_multi_trainingValidation.ipynb    # Builds, tunes, trains, validates, and exports ML models for multi-class classification
├── cnn_base.ipynb    # Builds, tunes, trains, validates, and exports base CNN for binary classification
├── cnn_multi.ipynb    # Builds, tunes, trains, validates, and exports standard CNN model for multi-class classification
└── requirments.txt/environment.yml          # Dependencies/Environment needed for this project
```

## Installation

### Creating & Activating a Conda Environment

1. **Install** [Anaconda/Miniconda](https://docs.conda.io/en/latest/) if you haven’t already.  
2. **Create** a new environment (example name: `edgeai`):  
   ```bash
   conda create -n iot_ids python=3.13.7
   ```
3. **Activate** it:  
   ```bash
   conda activate iot_ids
   ```

### Installing Dependencies

Within your **activated** conda environment:

```bash
conda install --file requirements.txt
```
OR
```bash
conda env create -f environment.yml
```

Or, if you prefer `pip`:

```bash
pip install -r requirements.txt
```

*(Make sure you are in the conda environment so packages are installed there.)*

**Typical Requirements** (already in `dependencies.txt`):
- Python 3.12 (or similar)
- PyTorch 2.8
- Pandas 2.3.3
- Matplotlib 3.10 (for plotting)
- scikit-learn 1.7.2 (for classification reports)
- imbalanced-learn 0.14.0 (for SMOTE class balancing)

---

## Usage

### 1. Data Preparation

**Manual download** required. 

dataPreprocessing.ipynb splits the dataset into **train** and **test** sets (80/20).

### 2. Model Training


### 3. Evaluation

---

## Results

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

We welcome suggestions, bug reports, and community contributions!

---

## License

This project is licensed under the [MIT License](LICENSE). You’re free to use, modify, and distribute the code as allowed by that license.

---

## References

1. **RT-IoT2022 Dataset** – B. S. and R. Nagapadma, “RT-IoT2022 ,” UCI Machine Learning Repository, 2023, DOI: https://doi.org/10.24432/C5P338.

---

_Thank you for visiting **CSI4999_IDSwithAI**! If you have any questions or issues, feel free to open an [issue](../../issues) or reach out._
