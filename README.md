
# XPER Clustering Experiments

This repository provides a modular and reproducible pipeline for **XPER-based clustering** and **feature-based clustering**, tailored to machine learning applications on heterogeneous datasets. It supports **performance decomposition**, **cluster-specific modeling**, and **comprehensive visual and statistical analysis** of results.

The code is written in **Python 3.12** and leverages key libraries such as `scikit-learn`, `sklearn-extra`, `xgboost`, `plotly`, and `pandas`. A complete list of dependencies is provided in [`requirements.txt`](#1-requirements).

---

## 📌 Table of Contents

- [1. Requirements](#1-requirements)
- [2. Getting Started](#2-getting-started)
- [3. Repository Structure](#3-repository-structure)
- [4. Main Scripts Overview](#4-main-scripts-overview)
- [5. Output & Experiment Logs](#5-output--experiment-logs)
- [6. Further Analysis](#6-further-analysis)

---

## 1. Requirements

- **Python 3.12**
- All dependencies listed in `requirements.txt`

### Installation

```bash
# Optionally create a new virtual environment
python3.12 -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## 2. Getting Started

Follow these steps to replicate or extend the core experiments:

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/xper-clustering.git
   cd xper-clustering
   ```

2. **Edit `config.py`** to set your experiment parameters:
   - `SAMPLE_SIZE` – number of rows sampled
   - `N_FEATURES` – number of features used
   - `KERNEL_USE` – toggle kernel approximation
   - Currently, only the **credit risk dataset** is supported.

3. **Run the main experiment pipeline**:
   ```bash
   python model_building.py
   ```

4. **Visualizations** will be generated automatically and saved within a timestamped `experiments/` directory.

5. To perform cluster-specific modeling and evaluation, execute:
   ```bash
   python model_building_cluster.py
   ```

6. To aggregate and compare results across runs, open:
   - `additional_analysis/aggregate_experiment_analysis.ipynb`

   This notebook creates:
   - `combined_analysis_frames.xlsx`
   - `test_combined_analysis_frames.xlsx`
   - `train_combined_analysis_frames.xlsx`

---

## 3. Repository Structure

```bash
.
├── additional_analysis/               # Advanced EDA and diagnostics
│   ├── baseline_analysis/             # Baseline model comparisons
│   ├── aggregate_experiment_analysis.ipynb
│   ├── eda_script.ipynb
│   ├── feature_vs_xper_outliers.ipynb
│   ├── features_distribution_by_default_class.ipynb
│   ├── heterogeneity.ipynb
│   └── train_test_distribution_consistency.ipynb
│
├── data/                              # (Optional) Raw and preprocessed data
├── experiments/                       # Auto-generated logs and results
├── XPER/                              # XPER logic and utils
│
├── config.py                          # Experiment configurations
├── load_data.py                       # Dataset loader
├── utils.py                           # Utility functions (e.g., model evaluator)
├── model_building.py                  # Global model pipeline
├── model_building_cluster.py         # Cluster-specific modeling pipeline
├── cluster_analysis.py                # Additional analysis on clustering results
├── visualizations.py                  # Global-level visualizations
├── visualizations_cluster.py          # Cluster-level visualizations
├── requirements.txt                   # Python dependencies
└── README.md                          # Project overview (you’re here!)
```

---

## 4. Main Scripts Overview

### `model_building.py`

- Core entry point for:
  - Data preprocessing and train/test splits
  - Model training (e.g., XGBoost)
  - XPER value computation (per-instance and global)
  - Feature-based and XPER-based K-Medoids clustering
  - Cluster assignment exports
  - Results aggregation and logging

### `model_building_cluster.py`

- Performs cluster-specific training and evaluation
- Outputs performance comparisons across clusters
- Stores results in a new subdirectory under `experiments/`

### `config.py`

- Central configuration hub:
  - Feature selection
  - Sample size
  - Experiment toggles
  - Dataset path or selection (single dataset currently supported)

---

## 5. Output & Experiment Logs

Each run creates a unique timestamped folder under `experiments/`, structured as follows:

```
experiments/experiment_YYYYMMDD_HHMMSS/
│
├── models/                      # Trained global or cluster models
├── data/                        # Train/test splits with metadata
├── xper_values/                 # Per-instance and global XPER values
├── visualizations/              # Cluster-specific and global plots
├── final_results.csv            # Metrics summary for this dataset
└── overall_results.csv          # Aggregated run-level metrics
```

Output files include:
- `train_xper_clusters.csv` / `test_xper_clusters.csv`
- `train_feature_clusters.csv` / `test_feature_clusters.csv`
- `train_per_instance_xper.csv` / `train_global_xper.csv`
- `train_test_distribution_stats.csv`
- `silhouette_scores.csv` (if clustering quality evaluation is enabled)

As the experiment folder is to big for me to push it to GitHub, I have opted to upload it to my Google Drive. Hence all results related to my thesis can be found [here](https://drive.google.com/drive/folders/1J2OVdf9u5IGflaEboVDVmuo8t2ZrRla5?usp=sharing)

---

## 6. Further Analysis

The `additional_analysis/` folder provides notebooks to analyze key phenomena such as:

- Distribution shifts between train and test sets
- Feature contributions vs. class labels
- Global vs. local attribution analysis
- Heterogeneity validation via statistical tests

Use these notebooks to:
- Interpret clustering quality
- Validate feature relevance
- Compare raw feature-based vs. XPER-based subgrouping