import os
import ast
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from loguru import logger

# ------------------------------------------------------------------------
# 1. Plotting function: XPER distribution boxplot
# ------------------------------------------------------------------------
def plot_xper_distribution(df_xper_values, save_path):
    """
    Plots a boxplot of XPER values for each feature (except 'Benchmark').
    Saves to 'xper_distribution.png' in save_path.
    """
    plt.figure(figsize=(12, 6))

    # Safely drop 'Benchmark' if it exists
    drop_cols = []
    if "Benchmark" in df_xper_values.columns:
        drop_cols.append("Benchmark")
    if "Unnamed: 0" in df_xper_values.columns:
        drop_cols.append("Unnamed: 0")

    sns.boxplot(data=df_xper_values.drop(columns=drop_cols), palette="Reds")
    plt.xticks(rotation=90)
    plt.title("Distribution of XPER Values Across Features")
    plt.xlabel("Features")
    plt.ylabel("XPER Value")

    out_file = os.path.join(save_path, "xper_distribution.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    logger.info(f"[INFO] Saved XPER distribution plot to {out_file}")

    # Uncomment below if you want interactive display:
    # plt.show()
    plt.close()


# ------------------------------------------------------------------------
# 2. Plotting function: 4x4 grid for feature distributions
# ------------------------------------------------------------------------
def plot_feature_distributions_grid(df_clusters, method, save_path):
    """
    Plots feature distributions in a 4x4 grid:
      - Histogram for continuous variables
      - Bar chart for categorical (less than 5 unique values)
      - Each cluster is colored distinctly
      - Manual legend for numeric features
      - Saves figure as 'feature_distributions_<method>.png'
      
    The subplots are ordered based on a desired feature order.
    """
    # Define the desired order of features.
    desired_order = ["Age", "Car price", "Job tenure", "Funding amount", "Down payment", "Loan duration"]
    
    # Get all features except for 'Index' and 'Cluster'
    features_unsorted = list(df_clusters.columns.difference(["Index", "Cluster"]))
    
    # Create an ordered list: features from desired_order (if they exist) followed by the rest sorted alphabetically.
    features_ordered = [feat for feat in desired_order if feat in features_unsorted]
    features_ordered += sorted([feat for feat in features_unsorted if feat not in desired_order])
    
    n_features = len(features_ordered)
    
    if n_features == 0:
        logger.info(f"[WARNING] No features to plot for {method}. Skipping.")
        return

    subplots = math.ceil(n_features / 4)
    fig, axes = plt.subplots(subplots, 4, figsize=(20, 16))
    axes = axes.flatten()

    # Distinct color palette based on the number of clusters.
    n_clusters = df_clusters["Cluster"].nunique()
    cluster_palette = sns.color_palette("tab10", n_colors=n_clusters)

    for i, feature in enumerate(features_ordered):
        ax = axes[i]
        # Replace typical "invalid" entries without inplace (to avoid warnings)
        df_clusters[feature] = df_clusters[feature].replace(["?", "None", "", np.inf, -np.inf], np.nan)

        # Drop NaNs in this feature
        df_clean = df_clusters.dropna(subset=[feature]).copy()

        # Attempt numeric conversion
        try:
            df_clean[feature] = pd.to_numeric(df_clean[feature], errors='coerce')
        except Exception as e:
            logger.warning(f"Conversion failed for feature {feature}: {e}")

        # After conversion, drop any newly introduced NaNs
        df_clean = df_clean.dropna(subset=[feature])

        if feature in ["Default (y)", "Down payment"]:
            # Categorical => Bar chart
            df_clean[feature] = df_clean[feature].astype(int, errors='ignore')
            category_counts = df_clean.groupby(["Cluster", feature]).size().reset_index(name="Count")
            total_counts = category_counts.groupby("Cluster")["Count"].transform("sum")
            category_counts["Percentage"] = category_counts["Count"] / total_counts * 100

            sns.barplot(
                data=category_counts,
                x=feature,
                y="Percentage",
                hue="Cluster",
                palette=cluster_palette,
                ax=ax
            )
            ax.set_ylabel("Percentage (%)")
            # Ensure x-ticks for all possible integer categories
            ax.set_xticks(sorted(category_counts[feature].unique()))
        else:
            # Continuous => Hist + KDE
            sns.histplot(
                data=df_clean,
                x=feature,
                hue="Cluster",
                kde=True,
                bins=30,
                palette=cluster_palette,
                alpha=0.6,
                ax=ax
            )
            ax.set_ylabel("Density")

            # Manual legend for numeric features
            cluster_labels = sorted(df_clean["Cluster"].unique())
            legend_patches = [
                mpatches.Patch(color=cluster_palette[idx], label=f"Cluster {cluster_val}")
                for idx, cluster_val in enumerate(cluster_labels)
            ]
            ax.legend(handles=legend_patches, title="Cluster", loc="upper right")

        ax.set_title(f"{feature} ({method})")
        ax.set_xlabel(feature)
        ax.tick_params(axis='x', rotation=45)

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Feature Distributions: {method}", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_file = os.path.join(save_path, f"feature_distributions_{method.replace(' ', '_')}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    logger.info(f"[INFO] Saved feature distributions for {method} to {out_file}")

    # plt.show()
    plt.close()

# ------------------------------------------------------------------------
# 3. Preprocessing function: standardize + remove outliers
# ------------------------------------------------------------------------
def preprocess_clusters(df_clusters):
    """
    Standardizes features (Z-score) and removes outliers beyond 3 SD.
    Returns a cleaned copy of df_clusters.
    """
    features = df_clusters.columns.difference(["Cluster", "Index"])
    scaler = StandardScaler()

    # Fit transform only on numeric columns
    numeric_cols = []
    for col in features:
        try:
            df_clusters[col] = pd.to_numeric(df_clusters[col])
            numeric_cols.append(col)
        except:
            pass

    df_clusters = df_clusters.dropna(subset=numeric_cols)

    df_clusters[numeric_cols] = scaler.fit_transform(df_clusters[numeric_cols])

    # Remove outliers
    mask = (df_clusters[numeric_cols].abs() < 3).all(axis=1)
    df_cleaned = df_clusters[mask].copy()
    removed = df_clusters.shape[0] - df_cleaned.shape[0]
    logger.info(f"[INFO] Removed {removed} outliers from {len(df_clusters)} total data points.")

    return df_cleaned


# ------------------------------------------------------------------------
# 4. PCA + Plotly scatter
# ------------------------------------------------------------------------
def plot_pca_clusters_plotly(df_clusters, method, save_path):
    """
    Applies 2-component PCA, then plots cluster separation with Plotly.
    Saves figure as 'pca_<method>.png' to save_path.
    """
    features = df_clusters.columns.difference(["Cluster", "Index"])
    # Convert to numeric, dropping non-numeric columns
    for col in features:
        df_clusters[col] = pd.to_numeric(df_clusters[col], errors='coerce')
    df_clusters = df_clusters.dropna(subset=features)

    X = df_clusters[features].values
    y = df_clusters["Cluster"].values

    if X.shape[1] < 1:
        logger.info(f"[WARNING] Not enough numeric features to run PCA for {method}. Skipping.")
        return

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_components = min(2, n_features, n_samples)

    if n_components < 2:
        logger.info(f"[WARNING] PCA cannot be performed with n_components < 2 (for {method}).")
        return

    pca = PCA(n_components=n_components, random_state=42)
    embedding = pca.fit_transform(X)

    df_pca = pd.DataFrame(embedding, columns=[f"PCA {i+1}" for i in range(n_components)])
    df_pca["Cluster"] = y.astype(str)

    fig = px.scatter(
        df_pca,
        x="PCA 1",
        y="PCA 2" if n_components > 1 else "PCA 1",
        color="Cluster",
        title=f"PCA Projection of {method} Clusters",
        template="plotly_white",
        width=1000,
        height=600
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_xaxes(tickangle=45)

    out_file = os.path.join(save_path, f"pca_{method.replace(' ', '_')}.png")
    try:
        fig.write_image(out_file)
        logger.info(f"[INFO] Saved PCA plot for {method} to {out_file}")
    except Exception as e:
        logger.info(f"[WARNING] Could not save Plotly figure. Install kaleido or orca. Error: {e}")


# ------------------------------------------------------------------------
# 5. Main loop: iterate over DATA_LIST, load data, process, visualize
# ------------------------------------------------------------------------
def main(BASE_DIR, RESULTS_FILE, DATA_LIST):
    """
    Main entry point for generating visualizations. Reads each dataset's cluster
    CSVs + full dataset, merges them, and plots distributions + PCA.
    """

    def ensure_dir(path):
        """Create directory if it doesn't exist."""
        if not os.path.exists(path):
            os.makedirs(path)

    logger.info(f"[INFO] RESULTS_FILE = {RESULTS_FILE}")
    if os.path.exists(RESULTS_FILE):
        overall_results = pd.read_csv(RESULTS_FILE)
        logger.info("[INFO] overall_results loaded successfully.")
    else:
        logger.info("[WARNING] overall_results file not found. Proceeding, but merges may fail.")
        overall_results = None

    logger.info("[INFO] DATA_LIST: %s", DATA_LIST)

    for dataset in DATA_LIST:
        for method in ["kmedoids", "kmeans", "gmm"]:
            dataset_name = method + "_" + dataset.lower().replace(" ", "_")
            logger.info(f"\n[INFO] Processing dataset: {dataset} ({dataset_name})")

            # Prepare directory for saving visualizations
            visualizations_path = os.path.join(BASE_DIR, dataset_name, "analysis")
            ensure_dir(visualizations_path)

            # File paths for cluster assignments (train/test, XPER/feature-based)
            test_feature_clusters_file = os.path.join(BASE_DIR, dataset_name, "xper_values", "test_feature_clusters.csv")
            test_xper_clusters_file = os.path.join(BASE_DIR, dataset_name, "xper_values", "test_xper_clusters.csv")
            test_error_clusters_file = os.path.join(BASE_DIR, dataset_name, "xper_values", "test_epsilon_clusters.csv")

            train_feature_clusters_file = os.path.join(BASE_DIR, dataset_name, "xper_values", "train_feature_clusters.csv")
            train_xper_clusters_file = os.path.join(BASE_DIR, dataset_name, "xper_values", "train_xper_clusters.csv")
            train_error_clusters_file = os.path.join(BASE_DIR, dataset_name, "xper_values", "train_xper_clusters.csv")

            # XPER values (test set)
            xper_values_file = os.path.join(BASE_DIR, dataset_name, "xper_values", "test_per_instance_xper.csv")

            # Original dataset (full)
            feature_data_file = os.path.join(BASE_DIR, dataset_name, "data", "full_dataset.csv")

            # Path to overall results
            path_results = os.path.join(BASE_DIR, dataset_name, "final_results.csv")

            # Check files exist
            required_files = [
                test_feature_clusters_file, test_xper_clusters_file, test_error_clusters_file,
                train_feature_clusters_file, train_xper_clusters_file, train_error_clusters_file,
                xper_values_file, feature_data_file, path_results
            ]
            if not all([os.path.exists(fp) for fp in required_files]):
                logger.info("[WARNING] One or more files missing for dataset: %s", dataset_name)
                continue

            # 1) Load CSVs (cluster assignments)
            test_df_feature_clusters = pd.read_csv(test_feature_clusters_file)
            test_df_xper_clusters = pd.read_csv(test_xper_clusters_file)
            test_df_error_clusters = pd.read_csv(test_error_clusters_file)

            train_df_feature_clusters = pd.read_csv(train_feature_clusters_file)
            train_df_xper_clusters = pd.read_csv(train_xper_clusters_file)
            train_df_error_clusters = pd.read_csv(train_error_clusters_file)

            # 2) Load XPER values (test set)
            df_xper_values = pd.read_csv(xper_values_file)

            # 3) Load full dataset with index_col=0 so the DF index matches the saved index
            df_features = pd.read_csv(feature_data_file, index_col=0)

            logger.info(f"[INFO] Loaded full dataset: {feature_data_file} with shape {df_features.shape}")

            # 4) Merge cluster assignments with original features by matching cluster "Index" to df_features.index
            #    Make sure test_df_feature_clusters only has 2 columns: [Index, Cluster].
            #    We'll do an inner join so we get rows that appear in both.
            test_df_feature_clusters = test_df_feature_clusters[["Index", "Cluster"]].merge(df_features, left_on="Index", right_index=True, how="inner")
            test_df_xper_clusters = test_df_xper_clusters[["Index", "Cluster"]].merge(df_features, left_on="Index", right_index=True, how="inner")
            test_df_error_clusters = test_df_error_clusters[["Index", "Cluster"]].merge(df_features, left_on="Index", right_index=True, how="inner")

            train_df_feature_clusters = train_df_feature_clusters.merge(df_features, left_on="Index", right_index=True, how="inner")
            train_df_xper_clusters = train_df_xper_clusters.merge(df_features, left_on="Index", right_index=True, how="inner")
            train_df_error_clusters = train_df_error_clusters.merge(df_features, left_on="Index", right_index=True, how="inner")

            # 5) Load overall results for cluster-level metrics
            df_results = pd.read_csv(path_results).drop(columns=['Unnamed: 0'], errors='ignore')
            logger.debug(path_results)

            # 6) Extract cluster results from df_results if needed
            try:
                test_xper_scores = ast.literal_eval(df_results["Test XPER Cluster Results"].values[0])
                test_feature_scores = ast.literal_eval(df_results["Test Feature Cluster Results"].values[0])
                test_epsilon_scores = ast.literal_eval(df_results["Test Error Cluster Results"].values[0])
                train_xper_scores = ast.literal_eval(df_results["Train XPER Cluster Results"].values[0])
                train_feature_scores = ast.literal_eval(df_results["Train Feature Cluster Results"].values[0])
                train_epsilon_scores = ast.literal_eval(df_results["Train Error Cluster Results"].values[0])
                baseline_eval_test = ast.literal_eval(df_results["Baseline Model Full Test Eval"].values[0])
                baseline_eval_train = ast.literal_eval(df_results["Baseline Model Full Train Eval"].values[0])

                df_test_xper_scores = pd.DataFrame.from_dict(test_xper_scores, orient='index')
                df_test_xper_scores['Data'] = 'test_xper_scores'

                df_test_feature_scores = pd.DataFrame.from_dict(test_feature_scores, orient='index')
                df_test_feature_scores['Data'] = 'test_feature_scores'

                df_test_epsilon_scores = pd.DataFrame.from_dict(test_epsilon_scores, orient='index')
                df_test_epsilon_scores['Data'] = 'test_epsilon_scores'

                df_train_xper_scores = pd.DataFrame.from_dict(train_xper_scores, orient='index')
                df_train_xper_scores['Data'] = 'train_xper_scores'

                df_train_feature_scores = pd.DataFrame.from_dict(train_feature_scores, orient='index')
                df_train_feature_scores['Data'] = 'train_feature_scores'

                df_train_epsilon_scores = pd.DataFrame.from_dict(train_epsilon_scores, orient='index')
                df_train_epsilon_scores['Data'] = 'train_epsilon_scores'

                baseline_eval_test['Data'] = 'test_baseline_eval'
                baseline_eval_train['Data'] = 'train_baseline_eval'

                df_eval = pd.concat([
                    df_train_xper_scores,
                    df_train_feature_scores,
                    df_train_epsilon_scores,
                    df_test_xper_scores,
                    df_test_feature_scores,
                    df_test_epsilon_scores
                ], ignore_index=True)
                logger.info("[INFO] Combined cluster-level metrics extracted.")
            except Exception as e:
                logger.info(f"[WARNING] Could not parse cluster result columns from overall_results: {e}")
                df_eval = None


            df_aggregates = df_eval.groupby("Data").apply(weighted_aggregator).reset_index()
            df_aggregates.loc[len(df_aggregates)] = baseline_eval_test
            df_aggregates.loc[len(df_aggregates)] = baseline_eval_train
            df_aggregates["Data"] = df_aggregates["Data"].apply(lambda x: f"weighted_average_{x}")

            analysis_path = os.path.join(visualizations_path, "analysis_frame.xlsx")
            pd.concat([df_eval, df_aggregates], axis=0).to_excel(analysis_path, index=False)
            logger.info(f"✅ Analysis Data saved to {analysis_path}")

            # ----------------------------------------------------------------
            # 7) Generate plots
            # ----------------------------------------------------------------

            # 7a) Plot distribution of XPER values (test set)
            plot_xper_distribution(df_xper_values, visualizations_path)

            # 7b) Plot 4x4 grid: XPER-based clusters (test/train)
            plot_feature_distributions_grid(test_df_xper_clusters, "Test XPER-Based Clustering", visualizations_path)
            plot_feature_distributions_grid(train_df_xper_clusters, "Train XPER-Based Clustering", visualizations_path)

            # 7c) Plot 4x4 grid: Feature-based clusters (test/train)
            plot_feature_distributions_grid(test_df_feature_clusters, "Test Feature-Based Clustering", visualizations_path)
            plot_feature_distributions_grid(train_df_feature_clusters, "Train Feature-Based Clustering", visualizations_path)

            # 7d) Plot 4x4 grid: Error-based clusters (test/train)
            plot_feature_distributions_grid(test_df_error_clusters, "Test Error-Based Clustering", visualizations_path)
            plot_feature_distributions_grid(train_df_error_clusters, "Train Error-Based Clustering", visualizations_path)

            # 8) Preprocess cluster data for PCA
            test_df_xper_clusters_clean = preprocess_clusters(test_df_xper_clusters)
            train_df_xper_clusters_clean = preprocess_clusters(train_df_xper_clusters)
            test_df_feature_clusters_clean = preprocess_clusters(test_df_feature_clusters)
            train_df_feature_clusters_clean = preprocess_clusters(train_df_feature_clusters)
            test_df_error_clusters_clean = preprocess_clusters(test_df_error_clusters)
            train_df_error_clusters_clean = preprocess_clusters(train_df_error_clusters)

            # 9) PCA scatter plots
            plot_pca_clusters_plotly(test_df_xper_clusters_clean, "Test XPER-Based Clustering", visualizations_path)
            plot_pca_clusters_plotly(train_df_xper_clusters_clean, "Train XPER-Based Clustering", visualizations_path)
            plot_pca_clusters_plotly(test_df_feature_clusters_clean, "Test Feature-Based Clustering", visualizations_path)
            plot_pca_clusters_plotly(train_df_feature_clusters_clean, "Train Feature-Based Clustering", visualizations_path)
            plot_pca_clusters_plotly(test_df_error_clusters_clean, "Test Error-Based Clustering", visualizations_path)
            plot_pca_clusters_plotly(train_df_error_clusters_clean, "Train Error-Based Clustering", visualizations_path)

            logger.info(f"[INFO] Finished processing dataset: {dataset}")

    logger.info("[INFO] All visualizations saved!")

