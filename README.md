
# XPER Clustering Experiments

This repository provides a complete pipeline for running **XPER-based** and **feature-based** clustering on various datasets, along with subsequent **visualization** and **analysis** of the results. It uses Python **3.12**, `sklearn_extra`, `plotly`, and several other libraries listed in [`requirements.txt`](#requirements).

Below is an overview of how to get started, the main entry points, and how the code is structured.

---

## Table of Contents

- [1. Requirements](#1-requirements)
- [2. Getting Started](#2-getting-started)
- [3. Project Structure](#3-project-structure)
  - [3.1. `model_building.py`](#31-model_buildingpy)
  - [3.2. `visualizations.py`](#32-visualizationspy)
  - [3.3. `config.py`](#33-configpy)
  - [3.4. `load_data.py` and `utils.py`](#34-load_datapy-and-utilspy)
  - [3.5. Experiments Directory](#35-experiments-directory)
- [4. Usage](#4-usage)
  - [4.1. Running an Experiment](#41-running-an-experiment)
  - [4.2. Visualizing Results](#42-visualizing-results)
- [5. Acknowledgements and Further Information](#5-acknowledgements-and-further-information)

---

## 1. Requirements

- **Python 3.12** (ensure your environment is up to date)
- All dependencies listed in `requirements.txt`

### Installing Requirements

```bash
# Create a fresh virtual environment (optional but recommended)
python3.12 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install all required libraries
pip install -r requirements.txt
```

---

## 2. Getting Started

1. **Clone this repository** (or download the source).  
2. Make sure your Python environment is **3.12** with all dependencies installed.  
3. From the repository **root** directory, you can run the **main experiment pipeline**:
   ```bash
   python model_building.py
   ```
   This executes the **model training**, **XPER computations**, **cluster assignments**, and **results** generation.  
4. After the models finish, **visualizations** are automatically generated and saved in the same run.  

---

## 3. Project Structure

Below is a brief description of the key files:

```
.
├── config.py
├── load_data.py
├── model_building.py
├── requirements.txt
├── utils.py
├── visualizations.py
├── data/
|   └── ...
└── experiments/
    └── experiment_results_YYYYMMDDHHMMSS
    └── ...
```

### 3.1. `model_building.py`

- **Entry Point** for the entire experiment pipeline.  
- Reads datasets via `load_data.py` and processes them:
  - Splits data into train/test.
  - Trains a baseline model (e.g., XGBoost).
  - Computes **XPER** values for train/test sets.
  - Performs both **XPER-based** and **feature-based** **K-Medoids** clustering.
  - Trains cluster-specific models.
  - Evaluates performance on test data.

- Produces:
  - **Models** saved to `<experiment_folder>/<dataset_name>/models/`
  - **Data** splits saved to `<experiment_folder>/<dataset_name>/data/`
  - **XPER** values/clusters saved to `<experiment_folder>/<dataset_name>/xper_values/`
  - A final CSV of results in `<experiment_folder>/<dataset_name>/final_results.csv`
  - A consolidated `overall_results.csv` at the root of `<experiment_folder>/`

### 3.2. `visualizations.py`

- Reads in the **cluster assignments** (train/test, XPER-based, feature-based) and merges them with the original dataset.  
- Generates:
  - **Boxplots** of XPER distributions.
  - **Feature distribution plots** for each cluster.
  - **PCA** scatter plots in 2D with color-coded clusters.
- Saves all plots in a `visualizations/` folder under each dataset’s experiment folder.

### 3.3. `config.py`

- Holds **experiment parameters** such as:
  - `SAMPLE_SIZE` – how many rows to sample per dataset (e.g., 500).
  - `N_FEATURES` – how many features to use from each dataset.
  - `DATA_LIST` – which datasets to run experiments on.
  - `RESULTS_FILE` – path to the final/overall results file (default: `overall_results.csv`).
- You can modify these parameters to control which datasets to process and how the experiments are conducted.

### 3.4. `load_data.py` and `utils.py`

- **`load_data.py`**  
  - Should define a function `load_datasets()` that returns a dictionary of `(dataset_name, (df, target_col))`.
  - This is where you integrate your custom data sources.

- **`utils.py`**  
  - Contains helper functions like `evaluate_model`, `initiate_model`, `identify_problem_type`, etc.
  - Streamlines repetitive tasks (e.g., classification/regression detection, scoring).

### 3.5. Experiments Directory

- **`experiment_results_YYYYMMDDHHMMSS/`** (auto-generated)  
  - Each run of `model_building.py` creates a unique time-stamped folder to store all results.
  - Subfolders for each dataset: 
    - `models/`  
    - `data/`  
    - `xper_values/`  
    - `visualizations/` (once the pipeline finishes and `visualizations.py` runs)

Inside those subfolders, you’ll find:

- **`full_dataset.csv`**: The entire dataset with the original row index.  
- **`train_xper_clusters.csv`** / **`test_xper_clusters.csv`**: Minimal cluster assignments (`Index`, `Cluster`) from XPER-based approach.  
- **`train_feature_clusters.csv`** / **`test_feature_clusters.csv`**: Minimal cluster assignments from feature-based approach.  
- **`train_per_instance_xper.csv`** / **`test_per_instance_xper.csv`**: Detailed per-instance XPER values.  
- **`train_global_xper.csv`** / **`test_global_xper.csv`**: Global XPER values across features.  
- **`final_results.csv`**: Summarizes all metrics for that dataset, including baseline scores, silhouette scores, etc.

---

## 4. Usage

### 4.1. Running an Experiment

1. **Edit `config.py`** to select which datasets to run. For example:
   ```python
   DATA_LIST = ["Bank Marketing", "Iris", "Some Other Dataset"]
   SAMPLE_SIZE = 500
   N_FEATURES = 6
   ...
   ```
2. **Launch** the experiment pipeline:
   ```bash
   python model_building.py
   ```
3. The pipeline will load each dataset from `load_data.py`, build models, generate XPER values, cluster them, and finally run `visualizations.main(...)` to produce plots.

4. Results appear under a newly created folder, e.g., `experiment_results_31012025195154/`.

### 4.2. Visualizing Results

- The code **automatically** calls `visualizations.main(...)` at the end of each run.  
- The generated images:
  - **XPER distribution**: Boxplot(s) in `visualizations/xper_distribution.png`.
  - **Feature distributions**: `feature_distributions_<method>.png`.
  - **PCA** scatter: `pca_<method>.png`.

You’ll also see logs indicating where each plot was saved.

---

## 5. Acknowledgements and Further Information

- **XPER**: We leverage `XPER.compute.Performance` from a custom library to compute explanatory feature contributions.  
- **Clustering**: We use **K-Medoids** from [`sklearn_extra`](https://github.com/scikit-learn-contrib/scikit-learn-extra).  
- **Plotting**: Primarily done with `seaborn`, `matplotlib`, and `plotly`.  
- **Modeling**: `XGBoost` or other scikit-learn models (through the `initiate_model` in `utils.py`).

If you have questions or need further clarification, feel free to open an issue or reach out. Happy experimenting!

---

## ToDo and Progress:

2. Benchmark to other papers approach
3. assess statificial significance of findings
4. Try finetuning all models but specifically benchmark model before
5. Data preprocessing (?)
6. Different Clustering Approaches
7. Different Models
8. Use more coalition values
9. If clusters are clean I have to fix the metrics
10. PCA Plots

- DONE (KMEANS) benchmark against epsilon (the errors i make). if i have one size fits all model. plot distribution of errors. split in 2-3 groups and cluster based on those 
- DONE work with probabilities not binary classification (no need for weighting)
- DONE play with true xper and kernel 