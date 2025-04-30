
# ============================================================================
# 1. IMPORTS
# ============================================================================

import os
import time
import joblib
import shutil
from loguru import logger
import ast
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

# Custom modules
from XPER.compute.Performance import ModelPerformance
from data.loader import load_datasets
from utils.utils import evaluate_model, initiate_model, identify_problem_type
import visualization.global_visualizations as visualizations_cluster

# ============================================================================
# 2. GLOBAL VARIABLES AND CONFIGURATION
# ============================================================================

# ONLY FOR CLUSTER ANALYSIS
EXPERIMENT_FOLDER = "experiment_results_29042025171022"


# Global directories (will be set later for each dataset)
model_dir = None
data_dir = None
xper_dir = None

# ============================================================================
# 3. ENVIRONMENT SETUP
# ============================================================================

BASE_DIR = f"experiments/cluster_{EXPERIMENT_FOLDER}"
os.environ["BASE_DIR"] = BASE_DIR

logger.info(f"Experiment base directory: {BASE_DIR}")

# Import configuration parameters from config.py
from config import (
    SAMPLE_SIZE, 
    N_FEATURES, 
    DATA_LIST, 
    RESULTS_FILE, 
    KERNEL_USE, 
    BOOTSTRAP, 
    N_BOOTSTRAP, 
    N_SAMPELS
    )

# ============================================================================
# 4. HELPER FUNCTIONS
# ============================================================================

def create_experiment_directories(folder_name: str) -> None:
    """
    Creates the necessary directory structure for each dataset experiment.
    This includes directories for models, data, and XPER values.
    """
    global model_dir, data_dir, xper_dir #TODO: Are they not set globally already?

    # Construct paths based on the BASE_DIR and dataset folder name.
    model_dir = os.path.join(BASE_DIR, folder_name, "models")
    data_dir = os.path.join(BASE_DIR, folder_name, "data")
    xper_dir = os.path.join(BASE_DIR, folder_name, "xper_values")

    # Create directories if they do not exist.
    for directory in [model_dir, data_dir, xper_dir]:
        os.makedirs(directory, exist_ok=True)


# ============================================================================
# 5. DATA PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
    sample_size: int = 500,
    n_features: int = 6
):
    """
    Preprocess the given dataframe:
      1. Optionally sample a subset of the data.
      2. Select the first n_features from the feature set.
      3. Save the full dataset (with original index) to CSV.
      4. Split the data into training and testing sets.
    """
    label_encoder = LabelEncoder()

    if df.shape[0] > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Select features and target. Use first n_features columns.
    X = df.drop(columns=[target_col]).iloc[:, :n_features]
    y = df[target_col]

    out_full_path = os.path.join(data_dir, "full_dataset.csv")
    X.merge(y.to_frame(), left_index=True, right_index=True).to_csv(out_full_path, index=True)

    # Split data into training and testing sets (keep original indices)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    # Encode categorical features if any exist.
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        encoder = OrdinalEncoder()
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

    # Identify problem type using a custom function.
    model_type, num_classes = identify_problem_type(
        dataset_name, y_train, y_test, target_col, label_encoder
    )

    return X_train, X_test, y_train, y_test, model_type, num_classes


# ============================================================================
# 7. MODULAR CLUSTERING METHODS
# ============================================================================

def apply_clustering_method(data: np.ndarray, method: str = "kmedoids", n_clusters: int = 3, random_state: int = 42):
    """
    Apply a clustering algorithm on the provided data (2D array) and return cluster labels.
    """
    if method.lower() == "kmedoids":
        clustering_model = KMedoids(n_clusters=n_clusters, random_state=random_state)
        labels = clustering_model.fit_predict(data)
    elif method.lower() == "kmeans":
        clustering_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = clustering_model.fit_predict(data)
    elif method.lower() == "gmm":
        clustering_model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        labels = clustering_model.fit_predict(data)
    else:
        raise ValueError(f"Unknown clustering method '{method}'.")
    
    return labels, clustering_model

# ============================================================================
# 8. XPER-SPECIFIC FUNCTIONS
# ============================================================================

def compute_xper_values_and_cluster(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    min_clusters: int = 2, 
    max_clusters: int = 5, 
    method: str = "kmedoids"
    
):
    """
    Compute XPER values for a given model and cluster the per-instance XPER values.
    Uses a kernel if specified via KERNEL_USE from config.
    
    Returns:
        tuple: (cluster_labels, best_n_clusters, best_silhouette_score, scaler)
    """

    phi_per_instance = pd.read_csv(os.path.join("experiments", EXPERIMENT_FOLDER, "credit_risk", "xper_values", "train_per_instance_xper.csv"), index_col=0).values

    # Evaluate silhouette score.
    scaler = StandardScaler()
    xper_scaled = scaler.fit_transform(phi_per_instance[:, 1:])

    best_score = -1
    best_n_clusters = min_clusters
    best_cluster_labels = None

    # We iterate over a range to find the best silhouette score.
    for n_clusters in range(min_clusters, max_clusters + 1):
        labels, cluster_model = apply_clustering_method(xper_scaled, method=method, n_clusters=n_clusters, random_state=3)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(xper_scaled, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_cluster_labels = labels
                joblib.dump(cluster_model, os.path.join(model_dir, "xper_cluster_model.pkl"))
    
    # Save clustering assignments.
    df_cluster_info = pd.DataFrame({
        "Index": X.index,
        "Cluster": best_cluster_labels
    })

    csv_path = os.path.join(xper_dir, "train_xper_clusters.csv")
    df_cluster_info.to_csv(csv_path, index=False)
    logger.info(f"✅ XPER clustering saved to {csv_path}")
    
    return best_cluster_labels, best_n_clusters, best_score, scaler

def save_xper_results(X: pd.DataFrame, phi_global_values: np.ndarray, phi_per_instance_values: np.ndarray) -> None:
    """
    Save global and per-instance XPER values to CSV files.
    """
    df_global_xper = pd.DataFrame(phi_global_values, columns=["Global XPER"])
    df_global_xper.to_csv(os.path.join(xper_dir, "train_global_xper.csv"), index=False)

    col_names = ["Benchmark"] + list(X.columns)
    df_per_instance_xper = pd.DataFrame(phi_per_instance_values, columns=col_names)
    # Ensure the index aligns with X.
    df_per_instance_xper.index = X.index  
    df_per_instance_xper.to_csv(os.path.join(xper_dir, "train_per_instance_xper.csv"), index=True)


# ============================================================================
# 9. FEATURE-BASED CLUSTERING FUNCTIONS
# ============================================================================

def cluster_data_by_features(X: pd.DataFrame, min_clusters: int = 2, max_clusters: int = 5, method: str = "kmedoids"):
    """
    Cluster the training data in the original feature space using a specified
    clustering algorithm (default: kmedoids).
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_score = -1
    best_n_clusters = min_clusters
    best_cluster_labels = None

    # We iterate over a range to find the best silhouette score.
    for n_clusters in range(min_clusters, max_clusters + 1):
        labels, cluster_model = apply_clustering_method(X_scaled, method=method, n_clusters=n_clusters, random_state=3)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_cluster_labels = labels
                joblib.dump(cluster_model, os.path.join(model_dir, "feature_cluster_model.pkl"))

    df_cluster_info = pd.DataFrame({
        "Index": X.index,
        "Cluster": best_cluster_labels
    })
    csv_path = os.path.join(xper_dir, "train_feature_clusters.csv")
    df_cluster_info.to_csv(csv_path, index=False)
    logger.info(f"✅ Feature-based clustering saved to {csv_path}")

    return best_cluster_labels, best_n_clusters, best_score, scaler

# ============================================================================
# 10. EPSILON-BASED CLUSTERING FUNCTIONS
# ============================================================================

def cluster_data_by_error(y_true: pd.Series, y_pred: pd.Series, min_clusters: int = 2, max_clusters: int = 5, method: str = "kmeans", random_state: int = 42) -> np.ndarray:
    """
    Cluster the samples based on the error (epsilon) values.
    The error is defined as the absolute difference between the true and predicted values.
    """
    errors = y_true - y_pred
    errors = errors.reshape(-1) if hasattr(errors, "reshape") else np.array(errors)

    scaler = StandardScaler()
    errors_scaled = scaler.fit_transform(errors.reshape(-1, 1))

    best_score = -1
    best_n_clusters = min_clusters
    best_cluster_labels = None

    # We iterate over a range to find the best silhouette score.
    for n_clusters in range(min_clusters, max_clusters + 1):
        labels, cluster_model = apply_clustering_method(errors_scaled, method=method, n_clusters=n_clusters, random_state=random_state)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(errors_scaled, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_cluster_labels = labels
                joblib.dump(cluster_model, os.path.join(model_dir, "error_cluster_model.pkl"))

    
    df_cluster_info = pd.DataFrame({
        "Index": y_true.index,
        "Cluster": best_cluster_labels
    })
    csv_path = os.path.join(xper_dir, "train_epsilon_clusters.csv")
    df_cluster_info.to_csv(csv_path, index=False)
    logger.info(f"✅ Error-based clustering saved to {csv_path}")

    return best_cluster_labels, best_n_clusters, best_score, scaler

# ============================================================================
# 11. BOOTSTRAP CONFIDENCE INTERVALS FOR XPER VALUES
# ============================================================================

def bootstrap_xper_confidence_intervals(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    metric: str = "AUC",
    n_bootstraps: int = 1000,
    n_samples: int = 500,
    alpha: float = 0.05,
    kernel: str = None
):

    bootstrapped_values = []
    
    # Bootstrap iterations
    for i in range(n_bootstraps):
        # Sample indices with replacement
        indices = np.random.choice(np.arange(X.shape[0]), size=n_samples, replace=True)
        X_boot = X.iloc[indices]
        y_boot = y.iloc[indices]
        
        # Compute XPER on the bootstrap sample
        try:
            boot_instance = ModelPerformance(X_boot.values, y_boot.values, X_boot.values, y_boot.values, model, sample_size=X_boot.shape[0])
            boot_global, _ = boot_instance.calculate_XPER_values([metric], kernel=kernel, execution_type="ProcessPoolExecutor")
            bootstrapped_values.append(boot_global) 
        except Exception as e:
            logger.warning(f"Bootstrap iteration {i} failed with error: {e}")
            continue

    bootstrapped_df = pd.DataFrame(np.concatenate(bootstrapped_values))
    bootstrapped_df.columns = ["Base"] + X.columns

    lower_bounds = []
    upper_bounds = []

    for col in bootstrapped_df.columns:
        bootstrapped_col = bootstrapped_df[col].to_numpy()
        lower_bound = np.percentile(bootstrapped_col, 100 * alpha / 2)
        upper_bound = np.percentile(bootstrapped_col, 100 * (1 - alpha / 2))
        logger.info(f"Bootstrap CI for global XPER ({col}): {lower_bound} - {upper_bound}")

        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
    
    return lower_bound, upper_bound

# ============================================================================
# 13. CLUSTER-SPECIFIC TRAINING AND EVALUATION FUNCTIONS
# ============================================================================
def train_and_evaluate_cluster_models(X_train, y_train, target_col, cluster_labels, model_prefix):
    """
    For each cluster in binary classification, train a new classifier on that cluster's data
    and compute evaluation metrics. If a cluster is pure (all samples have the same label),
    we compute metrics using a dummy prediction (constant output and probability).
    """
    # Create a local copy and attach the cluster labels.
    #TODO: I dont seem to have an issue - but double check if variables are not overwritten (pd)
    X_train_local = X_train.copy()
    X_train_local["Cluster"] = cluster_labels

    cluster_metrics = {}
    label_encoders = {}

    # Loop over each unique cluster.
    for cluster_id in np.unique(cluster_labels):
        cluster_indices = X_train_local[X_train_local["Cluster"] == cluster_id].index
        X_train_cluster = X_train_local.loc[cluster_indices].drop(columns=["Cluster"])
        y_train_cluster = y_train.loc[cluster_indices]

        # Ensure y_train_cluster is a DataFrame with the proper column name.
        if isinstance(y_train_cluster, pd.Series):
            y_train_cluster = y_train_cluster.to_frame(name=target_col)
        
        unique_classes = np.sort(y_train_cluster[target_col].unique())

        # If the cluster is pure (only one unique class), use dummy predictions.
        if len(unique_classes) == 1:
            dummy_class = unique_classes[0]
            # Create dummy predictions: all predicted as the constant class.
            y_pred = np.full(shape=y_train_cluster.shape[0], fill_value=dummy_class)
            # Set predicted probabilities accordingly.
            # For example, if dummy_class == 1, then probability = 1.0; if 0, then 0.0.
            y_proba = np.ones(shape=y_train_cluster.shape[0]) if dummy_class == 1 else np.zeros(shape=y_train_cluster.shape[0])
            # NOTE: This is necessary because certain metric functions require certain input formats (The same approach must not be true for test)
            # Compute metrics on these dummy predictions.
            metrics = _compute_test_metrics(y_train_cluster[target_col], y_pred, y_proba, cluster_id)
            cluster_metrics[cluster_id] = metrics
            
            # Track the pure cluster.
            pure_clusters_dict[model_prefix][str(cluster_id)] = dummy_class
            logger.info(f"{model_prefix.upper()} cluster {cluster_id} is pure with class {dummy_class}. Dummy predictions used.")
            continue

        # For non-pure clusters, perform label encoding.
        temp_label_encoder = LabelEncoder() #TODO: When would i actually ever need this? in a binary case either the cluster is pure or i have 0, 1 (in the case of multiclass i would have to pass the encoder to the test case (maybe also store it in a dic to for pure clusters)
        temp_label_encoder.fit(unique_classes)
        y_train_cluster_encoded = temp_label_encoder.transform(y_train_cluster[target_col])
        label_encoders[cluster_id] = temp_label_encoder

        # Initialize a model with basic overfitting prevention.
        cluster_model = initiate_model("binary", num_classes=2, X_train=X_train_cluster, y_train=y_train_cluster_encoded)
        model_path = os.path.join(model_dir, f"{model_prefix}_cluster_{cluster_id}.pkl")
        joblib.dump(cluster_model, model_path)

        # Save the cluster's data for reference.
        df_cluster_data = pd.concat([X_train_cluster, y_train_cluster], axis=1)
        cluster_data_path = os.path.join(data_dir, f"{model_prefix}_cluster_{cluster_id}.csv") #TODO: wrong folder
        df_cluster_data.to_csv(cluster_data_path, index=True)

        # Evaluate on the training cluster._compute_test_metrics
        y_pred_encoded = cluster_model.predict(X_train_cluster)
        y_pred = temp_label_encoder.inverse_transform(y_pred_encoded)
        # TODO: For multiclass this would be a larger problem. cant just assume the probability vector matches from initial prediction
        try:
            y_proba = cluster_model.predict_proba(X_train_cluster)[:, 1]
        except Exception:
            y_proba = None

        metrics = _compute_test_metrics(y_train_cluster[target_col], y_pred, y_proba, cluster_id)
        cluster_metrics[cluster_id] = metrics

    return cluster_metrics, label_encoders

# ============================================================================
# 14. TEST-TIME EVALUATION FUNCTIONS FOR CLUSTERS
# ============================================================================

def _load_all_models_in_directory(models_folder: str) -> dict:
    """
    Utility function to load all model files (.pkl) from a directory.
    Returns a dictionary with keys as file names (without extension).
    """
    loaded_models = {}
    for fname in os.listdir(models_folder):
        if fname.endswith(".pkl"):
            key = fname.split('.')[0]
            model_path = os.path.join(models_folder, fname)
            loaded_models[key] = joblib.load(model_path)
    return loaded_models

def _save_test_xper_cluster_labels(X_test, cluster_labels, phi_per_instance_test) -> None:
    """
    Save the cluster assignments for test data based on XPER clustering.
    """
    test_xper_data = phi_per_instance_test[:, 1:]
    df_test_clusters = pd.DataFrame({
        "Index": X_test.index,
        "Cluster": cluster_labels
    })
    xper_col_names = list(X_test.columns)
    df_xper_values = pd.DataFrame(test_xper_data, columns=xper_col_names)
    df_combined = pd.concat([df_test_clusters, df_xper_values], axis=1)
    output_path = os.path.join(xper_dir, "test_xper_clusters.csv")
    df_combined.to_csv(output_path, index=False)
    logger.info(f"✅ XPER clustering saved to {output_path}")

def _handle_pure_cluster(cluster_id, n_samples, model_prefix):
    """
    Handle the scenario where a cluster is pure (contains only one class).
    Returns predicted values and probabilities.
    """
    pure_class = pure_clusters_dict.get(model_prefix, {}).get(str(cluster_id))
    if pure_class is not None:
        logger.info(f"Detected pure cluster {cluster_id} ({model_prefix}). All test samples assigned to class {pure_class}.")
        if str(pure_class) == "1":
            predicted_proba = np.ones(n_samples, dtype=float)
        else:
            predicted_proba = np.zeros(n_samples, dtype=float)
        predicted_values = np.full(n_samples, pure_class)
        return predicted_values, predicted_proba

    logger.warning(f"Cluster {cluster_id} not found among pure clusters. Returning zeros.")
    return np.zeros(n_samples), np.zeros(n_samples)

def _compute_test_metrics(y_true, y_pred, y_proba, cluster_id):
    """
    Compute evaluation metrics for binary classification for a given cluster.
    For pure clusters (only one class in y_true), AUC is set to None.
    Other metrics (accuracy, log loss, Brier score, etc.) are computed if possible.
    """

    unique_labels = np.unique(y_true.values)
    is_pure_true = (len(unique_labels) == 1)

    unique_labels = np.unique(y_pred)
    is_pure_pred = (len(unique_labels) == 1)

    is_pure = is_pure_true and is_pure_pred
    
    if is_pure and (y_pred[0] == y_true.values[0]):
        auc_score = 1.0  # NOTE: This is not a perfectly clean solution.
    elif is_pure:
        auc_score = 0.0     
    else:
        auc_score = roc_auc_score(y_true, y_proba)

    logloss = log_loss(y_true.values, y_proba, labels=[0, 1])
    brier = brier_score_loss(y_true.values, y_proba)

    accuracy = accuracy_score(y_true.values, y_pred)
    
    recall = recall_score(y_true.values, y_pred) if is_pure == False else None
    f1 = f1_score(y_true.values, y_pred) if is_pure == False else None
    precision = precision_score(y_true.values, y_pred) if is_pure == False  else None


    return {
        "AUC Score": auc_score,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (TPR)": recall,
        "F1 Score": f1,
        "Log Loss": logloss,
        "Brier Score": brier,
        "Cluster Size": len(y_true),
        "Pure Cluster": True if is_pure else False,
        "Cluster ID": cluster_id
    }

def _save_test_xper_values(X_test, phi_global_test, phi_per_instance_test) -> None:
    """
    Save the test set XPER values to CSV files.
    """
    df_global_xper = pd.DataFrame(phi_global_test, columns=["Global XPER"])
    df_global_xper.to_csv(os.path.join(xper_dir, "test_global_xper.csv"), index=False)

    col_names = ["Benchmark"] + list(X_test.columns)
    df_per_instance_xper = pd.DataFrame(phi_per_instance_test, columns=col_names)
    df_per_instance_xper.index = X_test.index
    df_per_instance_xper.to_csv(os.path.join(xper_dir, "test_per_instance_xper.csv"), index=True)


def evaluate_test_data_xper_clusters(X_test, y_test, xper_scaler, label_encoders):
    """
    Evaluate test data using the XPER-based clustering approach.
    """
    models = _load_all_models_in_directory(model_dir)
    baseline_model = models.get("baseline_model")
    if baseline_model is None:
        logger.error("Baseline model not found. Cannot compute XPER on test data.")
        return {}

    phi_per_instance_test = pd.read_csv(os.path.join("experiments", EXPERIMENT_FOLDER, "credit_risk", "xper_values", "test_per_instance_xper.csv"), index_col=0).values

    col_names = ["Benchmark"] + list(X_test.columns)
    df_per_instance_xper = pd.DataFrame(phi_per_instance_test, columns=col_names)
    df_per_instance_xper.index = X_test.index
    df_per_instance_xper.to_csv(os.path.join(xper_dir, "test_per_instance_xper.csv"), index=True)

    xper_kmedoid_model = models.get("xper_cluster_model")
    if xper_kmedoid_model is None:
        logger.error("xper_cluster_model not found. Cannot cluster test data by XPER.")
        return {}

    test_xper_data = phi_per_instance_test[:, 1:]
    test_xper_data_scaled = xper_scaler.transform(test_xper_data)
    test_cluster_labels = xper_kmedoid_model.predict(test_xper_data_scaled)

    _save_test_xper_cluster_labels(X_test, test_cluster_labels, phi_per_instance_test)

    test_metrics = {}
    for cluster_id in np.unique(test_cluster_labels):
        cluster_indices = np.where(test_cluster_labels == cluster_id)[0]
        X_test_cluster = X_test.iloc[cluster_indices].copy()
        y_test_cluster = y_test.iloc[cluster_indices]

        model_key = f"xper_cluster_{cluster_id}"
        cluster_model = models.get(model_key)
        if cluster_model is None:
            predicted_values, predicted_proba = _handle_pure_cluster(cluster_id, X_test_cluster.shape[0], "xper")
        else:
            y_pred_encoded = cluster_model.predict(X_test_cluster)
            if cluster_id in label_encoders:
                le = label_encoders[cluster_id]
                y_pred_original = le.inverse_transform(y_pred_encoded)
            else:
                y_pred_original = y_pred_encoded

            y_pred_proba = cluster_model.predict_proba(X_test_cluster)[:, 1] # TODO: This again assumes binary classes. Thats the issue.
            predicted_values, predicted_proba = y_pred_original, y_pred_proba
            
        cluster_metrics = _compute_test_metrics(y_test_cluster, predicted_values, predicted_proba, cluster_id)
        test_metrics[cluster_id] = cluster_metrics

    return test_metrics

def evaluate_test_data_feature_clusters(X_test, y_test, feature_scaler, label_encoders):
    """
    Evaluate test data using the feature-based clustering approach.
    """
    models = _load_all_models_in_directory(model_dir)
    best_feature_kmedoid = models.get("feature_cluster_model")
    if best_feature_kmedoid is None:
        logger.error("feature_cluster_model not found. Cannot cluster test data by features.")
        return {}

    X_test_scaled = feature_scaler.transform(X_test)
    test_cluster_labels = best_feature_kmedoid.predict(X_test_scaled)

    X_test_local = X_test.copy()
    X_test_local["Cluster"] = test_cluster_labels
    X_test_local["Index"] = X_test.index
    feature_cluster_path = os.path.join(xper_dir, "test_feature_clusters.csv")
    X_test_local.to_csv(feature_cluster_path, index=False)
    logger.info(f"✅ Feature clustering saved to {feature_cluster_path}")

    test_metrics = {}
    for cluster_id in np.unique(test_cluster_labels):
        cluster_indices = X_test_local[X_test_local["Cluster"] == cluster_id].index
        X_test_cluster = X_test_local.loc[cluster_indices].drop(columns=["Cluster", "Index"])
        y_test_cluster = y_test.loc[cluster_indices]

        model_key = f"feature_cluster_{cluster_id}"
        cluster_model = models.get(model_key)
        if cluster_model is None:
            predicted_values, predicted_proba = _handle_pure_cluster(cluster_id, X_test_cluster.shape[0], "feature")
        else:
            y_pred_encoded = cluster_model.predict(X_test_cluster)
            if cluster_id in label_encoders:
                le = label_encoders[cluster_id]
                y_pred_original = le.inverse_transform(y_pred_encoded)
            else:
                y_pred_original = y_pred_encoded

            y_pred_proba = cluster_model.predict_proba(X_test_cluster)[:, 1]
            predicted_values, predicted_proba = y_pred_original, y_pred_proba
           
        cluster_metrics = _compute_test_metrics(y_test_cluster, predicted_values, predicted_proba, cluster_id)
        test_metrics[cluster_id] = cluster_metrics

    return test_metrics

def evaluate_test_data_epsilon_clusters(X_test, y_test, epsilon_scaler, label_encoders, baseline_model):
    """
    Evaluate test data using the epsilon (error)-based clustering approach.
    """

    models = _load_all_models_in_directory(model_dir)
    best_epsilon_model = models.get("error_cluster_model")
    if best_epsilon_model is None:
        logger.error("best_epsilon_cluster not found. Cannot cluster test data by errors.")
        return {}


    y_pred = baseline_model.predict_proba(X_test)[:, 1]
    errors = y_test - y_pred
    errors = errors.reshape(-1) if hasattr(errors, "reshape") else np.array(errors)

    errors_scaled = epsilon_scaler.transform(errors.reshape(-1, 1))
    test_cluster_labels = best_epsilon_model.predict(errors_scaled).astype(int).ravel()

    X_test_local = X_test.copy()
    X_test_local["Cluster"] = test_cluster_labels
    X_test_local["Index"] = X_test.index
    feature_cluster_path = os.path.join(xper_dir, "test_epsilon_clusters.csv")
    X_test_local.to_csv(feature_cluster_path, index=False)
    logger.info(f"✅ Error clustering saved to {feature_cluster_path}")

    test_metrics = {}
    for cluster_id in np.unique(test_cluster_labels):
        cluster_indices = X_test_local[X_test_local["Cluster"] == cluster_id].index
        X_test_cluster = X_test_local.loc[cluster_indices].drop(columns=["Cluster", "Index"])
        y_test_cluster = y_test.loc[cluster_indices]

        model_key = f"epsilon_cluster_{cluster_id}"
        cluster_model = models.get(model_key)
        if cluster_model is None:
            predicted_values, predicted_proba = _handle_pure_cluster(cluster_id, X_test_cluster.shape[0], "epsilon")
        else:
            y_pred_encoded = cluster_model.predict(X_test_cluster)
            if cluster_id in label_encoders:
                le = label_encoders[cluster_id]
                y_pred_original = le.inverse_transform(y_pred_encoded)
            else:
                y_pred_original = y_pred_encoded

            y_pred_proba = cluster_model.predict_proba(X_test_cluster)[:, 1]
            predicted_values, predicted_proba = y_pred_original, y_pred_proba
        
        cluster_metrics = _compute_test_metrics(y_test_cluster, predicted_values, predicted_proba, cluster_id)
        test_metrics[cluster_id] = cluster_metrics

    return test_metrics

# ============================================================================
# 15. VISUALIZATION FUNCTIONS (INCLUDING EPSILON DISTRIBUTION)
# ============================================================================

def plot_epsilon_violin(epsilon_train: np.ndarray, epsilon_test: np.ndarray, save_path: str):
    """
    Plot the distribution of epsilon (error) values for both training and test sets 
    using a violin plot. This method visualizes the density of the errors along with 
    key quartiles for each set.
    
    The function creates a DataFrame from the training and test error arrays, then 
    plots a side-by-side violin plot for easy comparison.
    """

    # Create a DataFrame in long format.
    df = pd.DataFrame({
        "Error": np.concatenate([epsilon_train, epsilon_test]),
        "Dataset": ["Train"] * len(epsilon_train) + ["Test"] * len(epsilon_test)
    })
    
    # Create the violin plot.
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Dataset", y="Error", data=df, palette={"Train": "blue", "Test": "red"}, inner="quartile")
    plt.title("Violin Plot of Epsilon (Error) Values (Train vs Test)")
    plt.xlabel("Dataset")
    plt.ylabel("Error")
    
    out_file = os.path.join(save_path, "epsilon_distribution_violin.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    logger.info(f"[INFO] Saved epsilon distribution violin plot to {out_file}")
    plt.close()

# ============================================================================
# 16. MAIN DATASET PROCESSING FUNCTION
# ============================================================================

def process_single_dataset(dataset_name: str, df: pd.DataFrame, target_col: str, method: str):

    logger.info(f"Processing dataset: {dataset_name}")
    start_time = time.time()

    X_train, X_test, y_train, y_test, model_type, num_classes = preprocess_data(
        df, target_col, dataset_name, sample_size=SAMPLE_SIZE, n_features=N_FEATURES
    )

    # Initialize baseline model with overfitting prevention.
    baseline_model = initiate_model(model_type, num_classes, X_train, y_train)
    joblib.dump(baseline_model, os.path.join(model_dir, "baseline_model.pkl"))

    baseline_score_train, baseline_score_test = evaluate_model(
        baseline_model, X_train, X_test, y_train, y_test, model_type
    )
    # Compute predictions for baseline test evaluation.
    y_test_proba = baseline_model.predict_proba(X_test)[:, 1]
    y_train_proba = baseline_model.predict_proba(X_train)[:, 1]
   
    y_pred_test = baseline_model.predict(X_test)
    y_pred_train = baseline_model.predict(X_train)

    baseline_eval_test = _compute_test_metrics(y_test, y_pred_test, y_test_proba, cluster_id="Baseline")
    baseline_eval_train = _compute_test_metrics(y_train, y_pred_train, y_train_proba, cluster_id="Baseline")
    logger.info(f"Baseline {model_type} - Train: {baseline_score_train}, Test: {baseline_score_test}")

    # Compute bootstrap confidence intervals for global XPER.

    if BOOTSTRAP:
        xper_lower, xper_upper = bootstrap_xper_confidence_intervals(
            X_train, y_train, baseline_model, metric="AUC", n_bootstraps=N_BOOTSTRAP, n_samples=N_SAMPELS, alpha=0.05, kernel=KERNEL_USE
        )

    # XPER-based clustering.
    xper_cluster_labels, xper_best_n_clusters, xper_best_score, xper_scaler = compute_xper_values_and_cluster(
        X_train, y_train, baseline_model, min_clusters=2, max_clusters=2, method=method
    )
    # NOTE: Max cluster =2 so only two clusters (no sillhuette comparison)

    # Feature-based clustering.
    feature_cluster_labels, feature_best_n_clusters, feature_best_score, feature_scaler = cluster_data_by_features(
        X_train, min_clusters=2, max_clusters=2, method=method
    )

    epsilon_train = y_train - y_train_proba
    epsilon_test = y_test - y_test_proba
    plot_epsilon_violin(epsilon_train, epsilon_test, xper_dir)

    # Epsilon-based clustering.
    # TODO: I think scaling is not consistent
    epsilon_cluster_labels, epsilon_best_n_clusters, epsilon_best_score, epsilon_scaler = cluster_data_by_error(
            y_train, y_pred=y_train_proba, min_clusters=2, max_clusters=2, method=method
    )

    # Train cluster-specific models.
    xper_cluster_results, xper_label_encoders = train_and_evaluate_cluster_models(
        X_train, y_train, target_col, xper_cluster_labels, "xper"
    )
    feature_cluster_results, feature_label_encoders = train_and_evaluate_cluster_models(
        X_train, y_train, target_col, feature_cluster_labels, "feature"
    )
    epsilon_cluster_results, epsilon_label_encoders = train_and_evaluate_cluster_models(
        X_train, y_train, target_col, epsilon_cluster_labels, "epsilon"
    )

    # Evaluate test data.
    test_xper_cluster_results = evaluate_test_data_xper_clusters(
        X_test, y_test, xper_scaler, xper_label_encoders
    )
    test_feature_cluster_results = evaluate_test_data_feature_clusters(
        X_test, y_test, feature_scaler, feature_label_encoders
    )
    test_epsilon_cluster_results = evaluate_test_data_epsilon_clusters(
        X_test, y_test, epsilon_scaler, epsilon_label_encoders, baseline_model
    )

    time_elapsed = round(time.time() - start_time, 2)
    results_summary = {
        "Dataset": dataset_name,
        "Model Type": model_type,
        "Train Sample Count": X_train.shape[0],
        "Train Feature Count": X_train.shape[1],
        "Baseline Model Train Score": baseline_score_train,
        "Baseline Model Test Score": baseline_score_test,
        "Baseline Model Full Test Eval": baseline_eval_test,
        "Baseline Model Full Train Eval": baseline_eval_train,
        "XPER-Based Cluster Count": xper_best_n_clusters,
        "XPER-Based Silhouette Score": xper_best_score,
        "Train XPER Cluster Results": xper_cluster_results,
        "Feature-Based Cluster Count": feature_best_n_clusters,
        "Feature-Based Silhouette Score": feature_best_score,
        "Train Feature Cluster Results": feature_cluster_results,
        "Train Error Cluster Results": epsilon_cluster_results,
        "Test XPER Cluster Results": test_xper_cluster_results,
        "Test Feature Cluster Results": test_feature_cluster_results,
        "Test Error Cluster Results": test_epsilon_cluster_results,
        "Computation Time (s)": time_elapsed
    }

    if BOOTSTRAP:
        results_summary["XPER CI Lower"] = xper_lower
        results_summary["XPER CI Upper"] = xper_upper

    return results_summary

# ============================================================================
# 17. EXPERIMENT RUNNER
# ============================================================================

def run_experiments() -> pd.DataFrame:
    """
    Loads multiple datasets, processes each one using the entire pipeline,
    and saves the results.
    Also copies the configuration file to the BASE_DIR for record-keeping.
    """
    global pure_clusters_dict

    all_datasets = load_datasets(DATA_LIST)
    all_results = []

    os.makedirs(BASE_DIR, exist_ok=True)
    shutil.copy('config.py', BASE_DIR)

    for dataset_name, (df, target_col) in all_datasets.items():
        if dataset_name in DATA_LIST:
            for method in ["kmedoids", "kmeans", "gmm"]:
                pure_clusters_dict = {"xper": {}, "feature": {}, "epsilon": {}}
                folder_name = method + "_" + dataset_name.lower().replace(" ", "_")
                create_experiment_directories(folder_name)
                result_dict = process_single_dataset(dataset_name, df, target_col, method)
                results_path = os.path.join(BASE_DIR, folder_name, "final_results.csv")
                pd.DataFrame([result_dict]).to_csv(results_path, index=False)
                all_results.append(result_dict)

    overall_results_df = pd.DataFrame(all_results)
    overall_results_path = os.path.join(BASE_DIR, folder_name, "overall_results.csv")
    overall_results_df.to_csv(overall_results_path, index=False)
    logger.info(f"Overall results saved to {overall_results_path}")

    return overall_results_df


# ============================================================================
# 18. ENTRY POINT
# ============================================================================

if __name__ == '__main__':

    final_results_df = run_experiments()
    
    logger.info("Experiment processing complete.")
    logger.info(BASE_DIR)
    
    visualizations_cluster.main(BASE_DIR, RESULTS_FILE, DATA_LIST)
    logger.info("Analysis processing complete.")
