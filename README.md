# IoT IDS — ML Transferability and Robustness

Cross-dataset benchmarking of ML/DL models for IoT intrusion detection.
Evaluates **transferability** and **robustness** of Random Forest, SVM, CNN, and LSTM across three public datasets using unified preprocessing and rigorous statistical analysis.

---

## Overview

| Property | Value |
| --- | --- |
| **Thesis title** | Cross-Dataset Transferability and Robustness of ML-Based IoT IDS |
| **Datasets** | Bot-IoT, UNSW-NB15, TON_IoT |
| **Models** | RF, SVM, CNN, LSTM |
| **Seeds** | 5 × random seeds (42, 101, 202, 303, 404) |
| **Primary metric** | Macro-averaged F1 |
| **Python** | 3.12 |
| **TensorFlow** | 2.21.0 |

---

## Repository Structure

```text
.
├── notebooks/
│   ├── config.py                   # shared imports, paths, constants
│   ├── preprocessing.ipynb         # Ch. 3.1  — clean, encode, split, scale
│   ├── model_training.ipynb        # Ch. 3.3  — train RF/SVM/CNN/LSTM, within-dataset eval
│   ├── transfer_evaluation.ipynb   # Ch. 3.4  — cross-dataset transfer eval
│   ├── stats_analysis.ipynb        # Ch. 3.5–3.7 — statistical tests, feature importance
│   ├── robustness_assessment.ipynb # Ch. 3.8  — 5-seed robustness
│   └── visualization.ipynb         # Ch. 4    — all figures
│
├── data/
│   ├── raw/                        # ← NOT in repo (download separately, see below)
│   ├── processed/                  # parquet files produced by preprocessing.ipynb
│   └── splits/                     # .npz train/test arrays + feature_list.json
│
├── models/                         # trained models (pkl / .keras)
├── results/                        # all CSV/TXT outputs
├── figures/                        # fig1–fig6 PNG
│
├── requirements.txt
└── README.md
```

---

## Dataset Download

Raw data is **not** included in this repository due to file size.
Download each dataset and place files exactly as shown:

| Dataset | Source | Place at |
| --- | --- | --- |
| **Bot-IoT** | UNSW Canberra IoT Lab | `data/raw/Bot-IoT/Bot-IoT_Dataset/Dataset/UNSW_2018_IoT_Botnet_Dataset_*.csv` |
| **UNSW-NB15** | UNSW Canberra | `data/raw/UNSW-NB15/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv` |
| **TON_IoT** | UNSW Canberra IoT Lab | `data/raw/TON_IoT/Train_Test_Network.csv` |

---

## Environment Setup

```bash
# requires Python 3.12
pip install -r requirements.txt
```

> **TensorFlow note:** `tensorflow==2.21.0` requires Python ≤ 3.12.

---

## Reproduction — Run Order

Run notebooks **in the order below**. Each notebook saves outputs to disk;
the next notebook loads them automatically via its first cell.

### Step 1 — `preprocessing.ipynb`

- Cleans raw CSVs, encodes labels, splits 80/20 (seed 42), fits StandardScaler
- **Saves to `data/splits/`:** `bot-iot.npz`, `unsw-nb15.npz`, `ton-iot.npz`, `feature_list.json`
- **Saves to `data/processed/`:** `bot_cl.parquet`, `unsw_cl.parquet`, `ton_cl.parquet`
- **Saves to `models/`:** `scaler_{dataset}.pkl`

### Step 2 — `model_training.ipynb`

- Trains RF, SVM, CNN, LSTM on each dataset (SMOTE on training sets only)
- Evaluates within-dataset performance; re-evaluates Bot-IoT at 100 k rows for consistency
- **Saves to `models/`:** `rf_*.pkl`, `svm_*.pkl`, `cnn_*_best.keras`, `lstm_*_best.keras`
- **Saves to `results/`:** `within_dataset_all.csv`

### Step 3 — `transfer_evaluation.ipynb`

- Loads saved models, evaluates each on the other two datasets (100 k-row cap)
- **Saves to `results/`:** `cross_dataset_all.csv`, `degradation_analysis.csv`, `ks_test_results.csv`, `bootstrap_ci.csv`, `transfer_classification.csv`

### Step 4 — `stats_analysis.ipynb`

- Friedman + Nemenyi post-hoc, pairwise Wilcoxon, OLS regression with VIF
- RF Gini importance, SVM permutation importance, CNN/LSTM SHAP
- **Saves to `results/`:** `pairwise_wilcoxon.csv`, `nemenyi_posthoc.csv`, `vif_results.csv`, `ols_summary.txt`, `rf_importance_*.csv`, `svm_permutation_*.csv`, `cnn_shap_*.csv`, `lstm_shap_*.csv`

### Step 5 — `robustness_assessment.ipynb`

- Re-trains RF and CNN across 5 seeds; reports variance in macro-F1
- **Saves to `results/`:** `robustness_all_seeds.csv`

### Step 6 — `visualization.ipynb`

- Generates all thesis figures from result CSVs
- **Saves to `figures/`:** `fig1_within_performance.png` through `fig6_robustness_boxplot.png`

---

## Key Results Summary

### Within-Dataset Performance (macro-F1)

| Model | Bot-IoT | UNSW-NB15 | TON_IoT |
| --- | --- | --- | --- |
| RF | 1.0000 | 0.9332 | 0.9947 |
| SVM | 0.5058 | 0.8712 | 0.8383 |
| CNN | 0.5376 | 0.9010 | 0.8596 |
| LSTM | 0.5525 | 0.9043 | 0.8856 |

### Cross-Dataset Transfer (macro-F1, selected)

| Source → Target | RF | SVM | CNN | LSTM |
| --- | --- | --- | --- | --- |
| Bot-IoT → UNSW-NB15 | 0.41 | 0.41 | 0.45 | 0.54 |
| UNSW-NB15 → Bot-IoT | 0.27 | 0.33 | 0.03 | 0.45 |
| TON_IoT → Bot-IoT | 0.00 | 0.00 | 0.00 | 0.00 |

**Transfer classification thresholds:** Total Failure (<0.01), Severe (0.01–0.20),
Poor (0.20–0.50), Moderate (0.50–0.75), Good (≥0.75)

---

## Figures

| File | Contents |
| --- | --- |
| `fig1_within_performance.png` | Within-dataset F1 / FPR bar chart |
| `fig2_cross_heatmap.png` | Cross-dataset transfer F1 heatmap |
| `fig3_degradation.png` | F1 degradation (within → cross) |
| `fig4_inference_time.png` | Inference latency by model/dataset |
| `fig5_ks_heatmap.png` | KS statistic feature-shift heatmap |
| `fig6_robustness_boxplot.png` | 5-seed F1 variance boxplot |

---

## Hardware Used

- CPU: 11th Gen Intel Core i5-1135G7 (4 cores / 8 threads, 2.42 GHz base / 4.2 GHz turbo)
- RAM: 8 GB DDR4
- GPU: Intel Iris Xe (integrated, limited TensorFlow acceleration)
- Storage: 477 GB SSD
- OS: Windows 11 Pro

---

## Citation / License

This repository accompanies a master's thesis. If you use the code or results,
please cite the thesis accordingly. Datasets are subject to their respective licenses
(see UNSW Canberra IoT Lab terms of use).
