# gene-expression-ai-pipeline

&#x20;

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Docker](#docker)
- [Usage](#usage)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Feature Selection](#2-feature-selection)
  - [3. Model Training](#3-model-training)
  - [4. Explainability](#4-explainability)
  - [Pipeline Orchestration](#pipeline-orchestration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project implements an end-to-end machine learning pipeline for predicting disease status from gene expression data. It covers data ingestion, preprocessing, feature selection, model training (Random Forest, XGBoost, Deep Neural Network), and SHAP-based explainability. The workflow is fully containerized with Docker for reproducibility.

## Features

- Automated loading and cleaning of GEO series matrix and family .soft files.
- Variance-based gene filtering via the elbow method.
- Random Forest-based feature importance selection.
- Hyperparameter tuning and training for RF, XGB, and DNN models.
- Model explainability with SHAP summary plots.
- Modular scripts and Jupyter notebooks in `src/` and `notebooks/`.
- Dockerized workflow with optional orchestration via `pipeline.py`.

## Directory Structure

```
.
├── data/
│   ├── GSE2034_series_matrix.txt.gz
│   └── GSE2034_family.soft.gz
├── notebooks/
│   ├── 01_data_loading_and_preprocessing.ipynb
│   ├── 02_feature_selection_randomForest.ipynb
│   ├── 03_randomForest.ipynb
│   ├── 04_xgboost.ipynb
│   ├── 05_dnn.ipynb
│   └── 06_shap_explainability.ipynb
├── src/
│   ├── preprocess.py
│   ├── feature_selection.py
│   ├── train_rf.py
│   ├── train_xgb.py
│   ├── train_dnn.py
│   └── explain_shap.py
├── figures/
├── models/
├── scalers/
├── reports/
├── pipeline.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- Docker (recommended for containerized runs)
- Git

### Clone the Repository

```bash
git clone https://github.com/chhuang216/gene-expression-ai-pipeline.git
cd gene-expression-ai-pipeline
```

### Python Environment (Local)

```bash
python -m venv env
source env/bin/activate      # Linux/macOS
env\Scripts\activate       # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### Docker

**Build the image:**

```bash
docker build -t chhuang216/gene-expression-ai-pipeline:latest .
```

**Pull from Docker Hub:**

```bash
docker pull chhuang216/gene-expression-ai-pipeline:latest
```

---

## Usage

### 1. Preprocessing

```bash
python src/preprocess.py \
  --series-matrix data/GSE2034_series_matrix.txt.gz \
  --family-soft data/GSE2034_family.soft.gz \
  --out-csv data/filtered_expression_data.csv \
  --plot-out figures/elbow_plot.png \
  --top-k 5000
```

### 2. Feature Selection

```bash
python src/feature_selection.py \
  --input-csv data/filtered_expression_data.csv \
  --k 20 \
  --out-csv data/selected_expression_data.csv \
  --plot-out figures/top20_importances.png
```

### 3. Model Training

#### Random Forest

```bash
python src/train_rf.py \
  --input-csv data/selected_expression_data.csv \
  --model-out models/rf_model.pkl \
  --scaler-out scalers/scaler_rf.pkl \
  --report-out reports/rf_report.txt \
  --test-size 0.2 --random-state 42
```

#### XGBoost

```bash
python src/train_xgb.py \
  --input-csv data/selected_expression_data.csv \
  --xgb-params '{"n_estimators":100,"max_depth":5}' \
  --model-out models/xgb_model.pkl \
  --scaler-out scalers/scaler_xgb.pkl \
  --report-out reports/xgb_report.txt \
  --test-size 0.2 --random-state 42
```

#### Deep Neural Network

```bash
python src/train_dnn.py \
  --input-csv data/selected_expression_data.csv \
  --model-out models/dnn_model.keras \
  --scaler-out scalers/scaler_dnn.pkl \
  --report-out reports/dnn_report.txt \
  --loss-plot-out figures/loss_plot.png \
  --matrix-plot-out figures/confusion_matrix.png \
  --test-size 0.2 --random-state 42 \
  --max-iter 500 --epochs 1000 --patient 5
```

### 4. Explainability (SHAP)

```bash
python src/explain_shap.py \
  --input-csv data/selected_expression_data.csv \
  --models-dir models \
  --scalers-dir scalers \
  --figures-dir figures \
  --test-size 0.2 --random-state 42
```

### Pipeline Orchestration

Run the full end-to-end workflow:

```bash
python pipeline.py
```

Or via Docker:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/scalers:/app/scalers \
  -v $(pwd)/figures:/app/figures \
  chhuang216/gene-expression-ai-pipeline:latest
```

---

## Results

- **Figures:** SHAP summary plots in `figures/`
- **Models:** Serialized classifiers in `models/`
- **Reports:** Performance metrics in `reports/`

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Chih-Hao (Andy) Huang – [chhuang216@gmail.com](mailto\:chhuang216@gmail.com)

