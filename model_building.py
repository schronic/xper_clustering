import os
import time
import joblib
import numpy as np
import pandas as pd
import shutil

from datetime import datetime
from loguru import logger

from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    silhouette_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from scipy.stats import ks_2samp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans


# Custom imports from your local modules
from XPER.compute.Performance import ModelPerformance
from load_data import load_datasets
from utils import evaluate_model, initiate_model, identify_problem_type
import visualizations



# ---------------------------------------------------
#            Environment Variable Setup
# ---------------------------------------------------

# Remove the existing BASE_DIR (if any) from the environment.
os.environ.pop("BASE_DIR", None)
logger.info("BASE_DIR reset. It will be regenerated on the next run.")

# Check if BASE_DIR is set; if not, create it with a timestamp.
if "BASE_DIR" in os.environ:
    BASE_DIR = os.environ["BASE_DIR"]
else:
    now = datetime.now()
    date_time = now.strftime("%d%m%Y%H%M%S")
    BASE_DIR = f"experiments/experiment_results_{date_time}"
    os.environ["BASE_DIR"] = BASE_DIR

logger.info(f"Experiment base directory: {BASE_DIR}")

from config import SAMPLE_SIZE, N_FEATURES, DATA_LIST, RESULTS_FILE, KERNEL_USE


# ---------------------------------------------------
#          Global or Shared Variables
# ---------------------------------------------------

model_dir = None
data_dir = None
xper_dir = None
#pure_clusters_dict = {"xper": {}, "feature": {}, "epsilon": {}}


# ---------------------------------------------------
#                Helper Functions
# ---------------------------------------------------

def create_experiment_directories(folder_name: str) -> None:
    """
    Creates the necessary directory structure for each dataset experiment.
    """
    global model_dir, data_dir, xper_dir, pure_clusters_dict
    
    # Reset dictionary that keeps track of pure clusters
    pure_clusters_dict = {"xper": {}, "feature": {}, "epsilon": {}}

    # Construct paths
    model_dir = os.path.join(BASE_DIR, folder_name, "models")
    data_dir = os.path.join(BASE_DIR, folder_name, "data")
    xper_dir = os.path.join(BASE_DIR, folder_name, "xper_values")

    # Create directories
    for directory in [model_dir, data_dir, xper_dir]:
        os.makedirs(directory, exist_ok=True)


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
    sample_size: int = 500,
    n_features: int = 6
):
    """
    Preprocess the given dataframe: sampling, encoding, splitting, and identifying the problem type.
    """
    label_encoder = LabelEncoder()

    # Limit dataset size
    if df.shape[0] > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        # <-- Do NOT reset_index(drop=True) here so we keep original index in place

    # Select features and target
    X = df.drop(columns=[target_col]).iloc[:, :n_features]
    y = df[target_col]

    # ------------------- NEW/CHANGED SECTION ---------------------------
    # Save the full dataset WITH index=True to preserve the original row index in the CSV
    dataset_dir = os.path.join(BASE_DIR, dataset_name.replace(" ", "_"))
    os.makedirs(dataset_dir, exist_ok=True)
    out_full_path = os.path.join(data_dir, "full_dataset.csv")
    # Store y first, then X, so you can see the target in col0 if you prefer, but index is crucial:
    X.merge(y.to_frame(), left_index=True, right_index=True).to_csv(out_full_path, index=True)
    # -------------------------------------------------------------------

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3
    )
    # <-- DO NOT reset_index(drop=True). We want to keep the original index

    # Encode categorical features if any
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        encoder = OrdinalEncoder()
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

    # Identify problem type
    model_type, classification_flag, num_classes = identify_problem_type(
        dataset_name, y_train, y_test, target_col, label_encoder
    )

    return X_train, X_test, y_train, y_test, model_type, num_classes, classification_flag


# ---------------------------------------------------
#           XPER-Specific Functions
# ---------------------------------------------------

def compute_xper_values_and_cluster(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    classification: bool
):
    """
    Compute XPER values for a given model, then cluster per-instance XPER values with KMedoids.
    """
    metric = "AUC" if classification else "R2"
    xper_instance = ModelPerformance(X.values, y.values, X.values, y.values, model, sample_size=X.shape[0])
    phi_global, phi_per_instance = xper_instance.calculate_XPER_values([metric], kernel=KERNEL_USE)

    # Save XPER values
    save_xper_results(X, phi_global, phi_per_instance)

    # Cluster by XPER
    cluster_labels, best_n_clusters, best_score, scaler = apply_kmedoids_clustering(X, phi_per_instance)
    return cluster_labels, best_n_clusters, best_score, scaler

def save_xper_results(X: pd.DataFrame, phi_global_values: np.ndarray, phi_per_instance_values: np.ndarray) -> None:
    """
    Save the computed XPER values to CSV files: global and per-instance.
    """
    df_global_xper = pd.DataFrame(phi_global_values, columns=["Global XPER"])
    df_global_xper.to_csv(
        os.path.join(xper_dir, "train_global_xper.csv"),
        index=False
    )

    col_names = ["Benchmark"] + list(X.columns)
    df_per_instance_xper = pd.DataFrame(phi_per_instance_values, columns=col_names)
    df_per_instance_xper.index = X.index  
    df_per_instance_xper.to_csv(
        os.path.join(xper_dir, "train_per_instance_xper.csv"),
        index=True  
    )

def apply_kmedoids_clustering(X: pd.DataFrame, phi_per_instance_values: np.ndarray, min_clusters: int = 2, max_clusters: int = 5):
    """
    Apply K-Medoids clustering on per-instance XPER values.
    """
    xper_data = phi_per_instance_values[:, 1:]
    scaler = StandardScaler()
    xper_data_scaled = scaler.fit_transform(xper_data)

    best_score = -1
    best_n_clusters = min_clusters
    best_kmedoid_model = None
    best_cluster_labels = None

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=3).fit(xper_data_scaled)
        labels = kmedoids.labels_
        if len(np.unique(labels)) > 1:
            score = silhouette_score(xper_data_scaled, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_kmedoid_model = kmedoids
                best_cluster_labels = labels

    joblib.dump(best_kmedoid_model, os.path.join(model_dir, "best_xper_kmedoid.pkl"))

    # ------------------- NEW/CHANGED SECTION ---------------------------
    # Instead of saving the entire feature set, just store Index + Cluster
    # This prevents duplicating columns when you read it back later.
    df_cluster_info = pd.DataFrame({
        "Index": X.index,     # original index
        "Cluster": best_cluster_labels
    })
    # We do not store the XPER values or the original features here; we keep it minimal
    csv_path = os.path.join(xper_dir, "train_xper_clusters.csv")
    df_cluster_info.to_csv(csv_path, index=False)
    # -------------------------------------------------------------------

    logger.info(f"✅ XPER clustering saved to {csv_path}")

    return best_cluster_labels, best_n_clusters, best_score, scaler


# ---------------------------------------------------
#          Feature-Based Clustering
# ---------------------------------------------------

def cluster_data_by_features(X: pd.DataFrame, min_clusters: int = 2, max_clusters: int = 5):
    """
    K-Medoids clustering on the original (scaled) feature space.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_score = -1
    best_n_clusters = min_clusters
    best_kmedoid_model = None
    best_cluster_labels = None

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=3).fit(X_scaled)
        labels = kmedoids.labels_
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_kmedoid_model = kmedoids
                best_cluster_labels = labels

    joblib.dump(best_kmedoid_model, os.path.join(model_dir, "best_feature_kmedoid.pkl"))

    # Store only Index + Cluster for the feature-based approach
    df_cluster_info = pd.DataFrame({
        "Index": X.index,
        "Cluster": best_cluster_labels
    })
    csv_path = os.path.join(xper_dir, "train_feature_clusters.csv")
    df_cluster_info.to_csv(csv_path, index=False)

    logger.info(f"✅ Feature-based clustering saved to {csv_path}")

    return best_cluster_labels, best_n_clusters, best_score, scaler


# ---------------------------------------------------
#          Epsilon-Based Clustering
# ---------------------------------------------------

def cluster_data_by_error(
    y_true: pd.Series, 
    y_pred: pd.Series, 
    n_clusters: int = 3,
    random_state: int = 42
) -> np.ndarray:

    #TODO: Save data and models

    errors = np.abs(y_true - y_pred)
    errors = errors.reshape(-1) if hasattr(errors, "reshape") else np.array(errors)

    cluster_model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = cluster_model.fit_predict(errors.reshape(-1, 1))

    joblib.dump(cluster_model, os.path.join(model_dir, "best_epsilon_cluster.pkl"))

    return labels

def cluster_data_by_error_wrapper(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model,   
    n_clusters: int = 3,
):
    
    y_pred = model.predict(X_train)
    labels = cluster_data_by_error(
        y_true=y_train, 
        y_pred=y_pred, 
        n_clusters=n_clusters, 
    )

    return labels


# ---------------------------------------------------
#       Cluster-Specific Training and Evaluation
# ---------------------------------------------------

def train_and_evaluate_cluster_models(X_train, y_train, target_col, cluster_labels, problem_type, model_prefix):
    """
    For each cluster, train a new model on that cluster's data.
    """
    global pure_clusters_dict
    X_train_local = X_train.copy()
    X_train_local["Cluster"] = cluster_labels

    cluster_metrics = {}
    label_encoders = {}

    for cluster_id in np.unique(cluster_labels):
        cluster_indices = X_train_local[X_train_local["Cluster"] == cluster_id].index
        X_train_cluster = X_train_local.loc[cluster_indices].drop(columns=["Cluster"])
        y_train_cluster = y_train.loc[cluster_indices]

        if isinstance(y_train_cluster, pd.Series):
            y_train_cluster = y_train_cluster.to_frame()
            y_train_cluster = y_train_cluster.rename(columns={y_train_cluster.columns[0]: target_col})

        unique_classes = np.sort(y_train_cluster[target_col].unique())

        if problem_type in ["binary", "multiclass"] and len(unique_classes) == 1:
            logger.info(
                f"{model_prefix.upper()} cluster {cluster_id} is purely class {unique_classes[0]}. Assigning perfect score."
            )
            """
            accuracy = accuracy_score(y_train_cluster_encoded, y_pred_binary)
            precision = precision_score(y_train_cluster_encoded, y_pred_binary, zero_division=0)
            recall = recall_score(y_train_cluster_encoded, y_pred_binary, zero_division=0)
            f1 = f1_score(y_train_cluster_encoded, y_pred_binary, zero_division=0)

            logloss = log_loss(y_train_cluster_encoded, y_proba)
            brier = brier_score_loss(y_train_cluster_encoded, y_proba)
            pos_proba = y_proba[y_train_cluster_encoded == 1]
            neg_proba = y_proba[y_train_cluster_encoded == 0]
            ks_statistic, _ = ks_2samp(pos_proba, neg_proba)


            cluster_metrics[cluster_id] = {
                "AUC Score": None,
                "Accuracy": None,
                "Precision": None,
                "Recall (TPR)": None,
                "F1 Score": None,
                "False Positive Rate (FPR)": None,
                "False Negative Rate (FNR)": None,
                "True Negative Rate (TNR)": None,
                "Log Loss": logloss,
                "Brier Score": brier,
                "KS Statistic": ks_statistic,
                "Cluster Size": len(cluster_indices),
                "Train Time (s)": "Pure Cluster",
            }


            pure_clusters_dict[model_prefix][str(cluster_id)] = unique_classes[0]
            continue
            """

        # Label Encode if classification
        temp_label_encoder = None
        if problem_type in ["binary", "multiclass"]:
            temp_label_encoder = LabelEncoder()
            temp_label_encoder.fit(unique_classes)
            y_train_cluster_encoded = temp_label_encoder.transform(y_train_cluster[target_col])
            label_encoders[cluster_id] = temp_label_encoder
        else:
            y_train_cluster_encoded = y_train_cluster[target_col].values

        start_time = time.time()
        if problem_type == "regression":
            cluster_model = initiate_model(problem_type)
        else:
            cluster_model = initiate_model(problem_type, len(unique_classes))

        cluster_model.fit(X_train_cluster, y_train_cluster_encoded)
        train_time = round(time.time() - start_time, 2)

        model_path = os.path.join(model_dir, f"{model_prefix}_cluster_{cluster_id}.pkl")
        joblib.dump(cluster_model, model_path)

        # Save cluster data for reference
        df_cluster_data = pd.concat([X_train_cluster, y_train_cluster], axis=1)
        df_cluster_data.to_csv(
            os.path.join(data_dir, f"{model_prefix}_cluster_{cluster_id}.csv"),
            index=True
        )

        # Evaluate on train set for reporting
        y_pred_encoded = cluster_model.predict(X_train_cluster)
        if temp_label_encoder:
            y_pred = temp_label_encoder.inverse_transform(y_pred_encoded)
        else:
            y_pred = y_pred_encoded

        if problem_type == "binary":
            y_proba = cluster_model.predict_proba(X_train_cluster)[:, 1]
            auc_score = roc_auc_score(y_train_cluster_encoded, y_proba)
            y_pred_binary = (y_proba > 0.5).astype(int)

            accuracy = accuracy_score(y_train_cluster_encoded, y_pred_binary)
            precision = precision_score(y_train_cluster_encoded, y_pred_binary, zero_division=0)
            recall = recall_score(y_train_cluster_encoded, y_pred_binary, zero_division=0)
            f1 = f1_score(y_train_cluster_encoded, y_pred_binary, zero_division=0)

            cm = confusion_matrix(y_train_cluster_encoded, y_pred_binary)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0


            logloss = log_loss(y_train_cluster_encoded, y_proba)
            brier = brier_score_loss(y_train_cluster_encoded, y_proba)
            pos_proba = y_proba[y_train_cluster_encoded == 1]
            neg_proba = y_proba[y_train_cluster_encoded == 0]
            ks_statistic, _ = ks_2samp(pos_proba, neg_proba)

            cluster_metrics[cluster_id] = {
                "AUC Score": auc_score,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall (TPR)": recall,
                "F1 Score": f1,
                "False Positive Rate (FPR)": fpr,
                "False Negative Rate (FNR)": fnr,
                "True Negative Rate (TNR)": tnr,
                "Log Loss": logloss,
                "Brier Score": brier,
                "KS Statistic": ks_statistic,
                "Cluster Size": len(cluster_indices),
                "Train Time (s)": train_time,
            }

        elif problem_type == "multiclass":
            accuracy = accuracy_score(y_train_cluster_encoded, y_pred_encoded)
            cluster_metrics[cluster_id] = {
                "Accuracy": accuracy,
                "Cluster Size": len(cluster_indices),
                "Train Time (s)": train_time,
            }
        else:  # regression
            mse = mean_squared_error(y_train_cluster_encoded, y_pred)
            cluster_metrics[cluster_id] = {
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "Cluster Size": len(cluster_indices),
                "Train Time (s)": train_time,
            }

    return cluster_metrics, label_encoders, pure_clusters_dict


# ---------------------------------------------------
#      Test-Time Evaluation on Clusters
# ---------------------------------------------------

def evaluate_test_data_xper_clusters(
    X_test, y_test, classification, xper_scaler, label_encoders
):
    """
    Predict on test data using XPER cluster approach.
    """
    models = _load_all_models_in_directory(model_dir)
    baseline_model = models.get("baseline_model")
    if baseline_model is None:
        logger.error("Baseline model not found. Cannot compute XPER on test data.")
        return {}

    metric = ["AUC"] if classification else ["R2"]
    xper_instance = ModelPerformance(X_test.values, y_test.values, X_test.values, y_test.values, baseline_model)
    phi_global_test, phi_per_instance_test = xper_instance.calculate_XPER_values(metric)

    _save_test_xper_values(X_test, phi_global_test, phi_per_instance_test)

    xper_kmedoid_model = models.get("best_xper_kmedoid")
    if xper_kmedoid_model is None:
        logger.error("best_xper_kmedoid not found. Cannot cluster test data by XPER.")
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

            if classification:
                y_pred_proba = cluster_model.predict_proba(X_test_cluster)[:, 1]
                predicted_values, predicted_proba = y_pred_original, y_pred_proba
            else:
                predicted_values, predicted_proba = y_pred_original, None

        cluster_metrics = _compute_test_metrics(y_test_cluster, predicted_values, predicted_proba, classification)
        test_metrics[cluster_id] = cluster_metrics

    return test_metrics

def evaluate_test_data_feature_clusters(
    X_test, y_test, classification, feature_scaler, label_encoders
):
    """
    Predict on test data using feature-based cluster approach.
    """
    models = _load_all_models_in_directory(model_dir)
    best_feature_kmedoid = models.get("best_feature_kmedoid")
    if best_feature_kmedoid is None:
        logger.error("best_feature_kmedoid not found. Cannot cluster test data by features.")
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

            if classification:
                y_pred_proba = cluster_model.predict_proba(X_test_cluster)[:, 1]
                predicted_values, predicted_proba = y_pred_original, y_pred_proba
            else:
                predicted_values, predicted_proba = y_pred_original, None

        cluster_metrics = _compute_test_metrics(y_test_cluster, predicted_values, predicted_proba, classification)
        test_metrics[cluster_id] = cluster_metrics

    return test_metrics

def evaluate_test_data_epsilon_clusters(
    X_test, y_test, classification, label_encoders, baseline_model
):
    
    models = _load_all_models_in_directory(model_dir)
    best_epsilon_model = models.get("best_epsilon_cluster")
    if best_epsilon_model is None:
        logger.error("best_feature_kmedoid not found. Cannot cluster test data by errors.")
        return {}
    
    y_pred = baseline_model.predict(X_test)
    
    errors = np.abs(y_test - y_pred)
    errors = errors.reshape(-1) if hasattr(errors, "reshape") else np.array(errors)
    test_cluster_labels = best_epsilon_model.predict(errors.reshape(-1, 1)).astype(int).ravel()

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

            if classification:
                y_pred_proba = cluster_model.predict_proba(X_test_cluster)[:, 1]
                predicted_values, predicted_proba = y_pred_original, y_pred_proba
            else:
                predicted_values, predicted_proba = y_pred_original, None

        cluster_metrics = _compute_test_metrics(y_test_cluster, predicted_values, predicted_proba, classification)
        test_metrics[cluster_id] = cluster_metrics

    return test_metrics


# ---------------------------------------------------
#         Internal Utility Functions
# ---------------------------------------------------

def _load_all_models_in_directory(models_folder: str) -> dict:
    loaded_models = {}
    for fname in os.listdir(models_folder):
        if fname.endswith(".pkl"):
            key = fname.split('.')[0]
            model_path = os.path.join(models_folder, fname)
            loaded_models[key] = joblib.load(model_path)
    return loaded_models


def _save_test_xper_values(X_test, phi_global_test, phi_per_instance_test) -> None:

    df_global_xper = pd.DataFrame(phi_global_test, columns=["Global XPER"])
    df_global_xper.to_csv(
        os.path.join(xper_dir, "test_global_xper.csv"),
        index=False
    )

    col_names = ["Benchmark"] + list(X_test.columns)
    df_per_instance_xper = pd.DataFrame(phi_per_instance_test, columns=col_names)
    df_per_instance_xper.index = X_test.index  
    df_per_instance_xper.to_csv(
        os.path.join(xper_dir, "test_per_instance_xper.csv"),
        index=True  
    )

def _save_test_xper_cluster_labels(X_test, cluster_labels, phi_per_instance_test) -> None:
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
    global pure_clusters_dict
    pure_class = pure_clusters_dict.get(model_prefix, {}).get(str(cluster_id))
    if pure_class is not None:
        logger.info(
            f"Detected pure cluster {cluster_id} ({model_prefix}). All test samples assigned to class {pure_class}."
        )
        if str(pure_class) == "1":
            predicted_proba = np.ones(n_samples, dtype=float)
        else:
            predicted_proba = np.zeros(n_samples, dtype=float)
        predicted_values = np.full(n_samples, pure_class)
        return predicted_values, predicted_proba

    logger.warning(f"Cluster {cluster_id} not found among pure clusters. Returning zeros.")
    return np.zeros(n_samples), np.zeros(n_samples)


def _compute_test_metrics(y_true, y_pred, y_proba, classification):
    if classification:
        unique_labels = np.unique(y_true.values)
        if len(unique_labels) != 2:
            logger.info(
                f"Single unique class in test cluster: {unique_labels}. Cannot compute AUC. Setting AUC=None."
            )
            auc_score = None
            accuracy = None
            fpr, fnr, tnr = None, None, None
            
        else:
            if y_proba is not None:
                auc_score = roc_auc_score(y_true.values, y_proba)
            else:
                auc_score = None
            cm = confusion_matrix(y_true.values, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                tn = fp = fn = tp = None
                fpr = fnr = tnr = None

            pos_proba = y_proba[y_true.values == 1]
            neg_proba = y_proba[y_true.values == 0]
            ks_statistic, _ = ks_2samp(pos_proba, neg_proba)

        accuracy = accuracy_score(y_true.values, y_pred)
        logloss = log_loss(y_true.values, y_proba)
        brier = brier_score_loss(y_true.values, y_proba)

        precision = precision_score(y_true.values, y_pred, zero_division=0)
        recall = recall_score(y_true.values, y_pred, zero_division=0)
        f1 = f1_score(y_true.values, y_pred, zero_division=0)

        return {
                "AUC Score": auc_score,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall (TPR)": recall,
                "F1 Score": f1,
                "False Positive Rate (FPR)": fpr,
                "False Negative Rate (FNR)": fnr,
                "True Negative Rate (TNR)": tnr,
                "Log Loss": logloss,
                "Brier Score": brier,
                "KS Statistic": ks_statistic,
                "Cluster Size": len(y_true),
            }
    
    # Regression
    mse = mean_squared_error(y_true.values, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "Cluster Size": len(y_true),
    }



# ---------------------------------------------------
#        Main Dataset Processing Function
# ---------------------------------------------------

def process_single_dataset(dataset_name: str, df: pd.DataFrame, target_col: str):
    """
    Execute the entire pipeline for a single dataset:
    1. Preprocess data
    2. Train baseline model
    3. XPER-based cluster (train)
    4. Feature-based cluster (train)
    5. Train cluster models
    6. Evaluate on test set
    7. Summarize results
    """
    logger.info(f"Processing dataset: {dataset_name}")
    start_time = time.time()

    X_train, X_test, y_train, y_test, model_type, num_classes, is_classification = preprocess_data(
        df, target_col, dataset_name, sample_size=SAMPLE_SIZE, n_features=N_FEATURES
    )

    # Baseline model
    baseline_model = initiate_model(model_type, num_classes)
    baseline_model.fit(X_train, y_train)
    joblib.dump(baseline_model, os.path.join(model_dir, "baseline_model.pkl"))

    baseline_score_train, baseline_score_test = evaluate_model(
        baseline_model, X_train, X_test, y_train, y_test, model_type
    )
    y_proba = baseline_model.predict_proba(X_test)[:, 1]
    y_pred = baseline_model.predict(X_test)
    baseline_eval =  _compute_test_metrics(y_test, y_pred, y_proba, is_classification)

    logger.info(f"Baseline {model_type} - Train: {baseline_score_train}, Test: {baseline_score_test}")

    # XPER-based clustering
    xper_cluster_labels, xper_best_n_clusters, xper_best_score, xper_scaler = compute_xper_values_and_cluster(
        X_train, y_train, baseline_model, is_classification
    )

    # Feature-based clustering
    feature_cluster_labels, feature_best_n_clusters, feature_best_score, feature_scaler = cluster_data_by_features(
        X_train
    )

    # Epsilon-based clustering
    epsilon_cluster_labels = cluster_data_by_error_wrapper(
        X_train,
        y_train,
        model = baseline_model,   
        n_clusters = 5,
    )

    # Train cluster models (XPER)
    xper_cluster_results, xper_label_encoders, _ = train_and_evaluate_cluster_models(
        X_train, y_train, target_col, xper_cluster_labels, model_type, "xper"
    )
    # Train cluster models (Feature)
    feature_cluster_results, feature_label_encoders, _ = train_and_evaluate_cluster_models(
        X_train, y_train, target_col, feature_cluster_labels, model_type, "feature"
    )
    # Train cluster models (Epsilon)
    epsilon_cluster_results, epsilon_label_encoders, _ = train_and_evaluate_cluster_models(
        X_train, y_train, target_col, epsilon_cluster_labels, model_type, "epsilon"
    )

    # Evaluate test data
    test_xper_cluster_results = evaluate_test_data_xper_clusters(
        X_test, y_test, is_classification, xper_scaler, xper_label_encoders
    )
    test_feature_cluster_results = evaluate_test_data_feature_clusters(
        X_test, y_test, is_classification, feature_scaler, feature_label_encoders
    )
    # Evaluate test data
    test_epsilson_cluster_results = evaluate_test_data_epsilon_clusters(
        X_test, y_test, is_classification, epsilon_label_encoders, baseline_model
    )

    time_elapsed = round(time.time() - start_time, 2)
    return {
        "Dataset": dataset_name,
        "Model Type": model_type,
        "Train Sample Count": X_train.shape[0],
        "Train Feature Count": X_train.shape[1],
        "Baseline Model Train Score": baseline_score_train,
        "Baseline Model Test Score": baseline_score_test,
        "Baseline Model Full Test Eval": baseline_eval,
        "XPER-Based Cluster Count": xper_best_n_clusters,
        "XPER-Based Silhouette Score": xper_best_score,
        "Train XPER Cluster Results": xper_cluster_results,
        "Feature-Based Cluster Count": feature_best_n_clusters,
        "Feature-Based Silhouette Score": feature_best_score,
        "Train Feature Cluster Results": feature_cluster_results,
        "Train Error Cluster Results": epsilon_cluster_results,
        "Test XPER Cluster Results": test_xper_cluster_results,
        "Test Feature Cluster Results": test_feature_cluster_results,
        "Test Error Cluster Results": test_epsilson_cluster_results,
        "Computation Time (s)": time_elapsed,
    }


def run_experiments() -> pd.DataFrame:
    """
    Loads multiple datasets, processes each one, and saves results.
    """
    all_datasets = load_datasets(DATA_LIST)
    all_results = []

    os.makedirs(BASE_DIR, exist_ok=True)
    shutil.copy('config.py', BASE_DIR)

    for dataset_name, (df, target_col) in all_datasets.items():

        if dataset_name in DATA_LIST:  # adjust if you want to run everything
            
            folder_name = dataset_name.lower().replace(" ", "_")
            create_experiment_directories(folder_name)

            result_dict = process_single_dataset(dataset_name, df, target_col)

            results_path = os.path.join(BASE_DIR, folder_name, "final_results.csv")
            pd.DataFrame([result_dict]).to_csv(results_path, index=False)
            
            all_results.append(result_dict)

    overall_results_df = pd.DataFrame(all_results)
    overall_results_path = os.path.join(BASE_DIR, "overall_results.csv")
    overall_results_df.to_csv(overall_results_path, index=False)
    
    logger.info(f"Overall results saved to {overall_results_path}")

    return overall_results_df


if __name__ == '__main__':
    final_results_df = run_experiments()
    logger.info("Experiment processing complete.")
    
    visualizations.main(BASE_DIR, RESULTS_FILE, DATA_LIST)
    logger.info("Analysis processing complete.")
