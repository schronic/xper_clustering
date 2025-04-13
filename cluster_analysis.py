
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from XPER import ModelPerformance




# Global configuration and random seed for reproducibility.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_EXPERIMENT_DIR = os.path.join(BASE_DIR, 'experiments')


def load_experiment_data(exp_path, cluster):

    data = {}

    train_clusters_file = os.path.join(exp_path, 'xper_values', 'train_xper_clusters.csv')
    test_clusters_file = os.path.join(exp_path, 'xper_values', 'test_xper_clusters.csv')

    xper_clusters_train = pd.read_csv(train_clusters_file, index_col=0).reset_index()
    xper_clusters_test = pd.read_csv(test_clusters_file, index_col=0).reset_index()

    # Load full dataset and split.
    data_path = os.path.join(exp_path, 'data', 'full_dataset.csv')
    df = pd.read_csv(data_path, index_col=0)
    X = df.drop(columns=["Default (y)"])
    y = df["Default (y)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    train_indices = xper_clusters_train.loc[xper_clusters_train['Cluster'] == int(cluster), 'Index'].values
    X_train = X_train.loc[train_indices]
    y_train = y_train.loc[train_indices]

    test_indices = xper_clusters_test.loc[xper_clusters_test['Cluster'] == int(cluster), 'Index'].values
    X_test = X_test.loc[test_indices]
    y_test = y_test.loc[test_indices]

    data['X_train'] = X_train
    data['X_test'] = X_test
    data['y_train'] = y_train
    data['y_test'] = y_test


    # Load baseline model.
    cluster_model_path = os.path.join(exp_path, 'models', f'xper_cluster_{str(cluster)}.pkl')
    cluster_model = joblib.load(cluster_model_path)

    if os.path.exists(os.path.join(exp_path, 'xper_values', f"cluster_{cluster}_xper_train.csv")):
        data['individual_train_xper'] = pd.read_csv(os.path.join(exp_path, 'xper_values', f"cluster_{cluster}_xper_train.csv"), index_col=0)
        data['global_train_xper'] = pd.read_csv(os.path.join(exp_path, 'xper_values', f"cluster_{cluster}_global_xper_train.csv"), index_col=0)

    else:
        xper_instance = ModelPerformance(X_train.values, y_train.values, X_train.values, y_train.values, cluster_model, sample_size=X_train.shape[0])
        phi_global_train, phi_per_instance_values = xper_instance.calculate_XPER_values(["AUC"], kernel=False) #TODO: Load from config

        col_names = ["Benchmark"] + list(X.columns)
        individual_train_xper = pd.DataFrame(phi_per_instance_values, columns=col_names)
        individual_train_xper.index = X_train.index  
        individual_train_xper.to_csv(os.path.join(exp_path, 'xper_values', f"cluster_{cluster}_xper_train.csv"), index=True)

        merged_train_xper = xper_clusters_train.merge(individual_train_xper, how='inner', left_index=True, right_index=True)
        data['individual_train_xper'] = merged_train_xper

        df_global_xper = pd.DataFrame(phi_global_train, columns=["Global XPER"])
        data['global_train_xper'] = df_global_xper
        df_global_xper.to_csv(os.path.join(exp_path, 'xper_values', f"cluster_{cluster}_global_xper_train.csv"), index=True)


    # TODO: use this opportunity to visualize
    if os.path.exists(os.path.join(exp_path, 'xper_values', f"cluster_{cluster}_xper_test.csv")):
        data['individual_test_xper'] = pd.read_csv(os.path.join(exp_path, 'xper_values', f"cluster_{cluster}_xper_test.csv"), index_col=0)
        data['global_test_xper'] = pd.read_csv(os.path.join(exp_path, 'xper_values', f"cluster_{cluster}_global_xper_test.csv"), index_col=0)

    else:

        xper_instance = ModelPerformance(X_test.values, y_test.values, X_test.values, y_test.values, cluster_model, sample_size=X_test.shape[0])
        phi_global_test, phi_per_instance_values = xper_instance.calculate_XPER_values(["AUC"], kernel=False) #TODO: Load from config

        col_names = ["Benchmark"] + list(X.columns)
        individual_test_xper = pd.DataFrame(phi_per_instance_values, columns=col_names)
        individual_test_xper.index = X_test.index  
        individual_test_xper.to_csv(os.path.join(exp_path, 'xper_values', f"cluster_{cluster}_xper_test.csv"), index=True)

        merged_test_xper = xper_clusters_test.merge(individual_test_xper, how='inner', left_index=True, right_index=True)
        data['individual_test_xper'] = merged_test_xper

        df_global_xper = pd.DataFrame(phi_global_test, columns=["Global XPER"])
        data['global_test_xper'] = df_global_xper
        df_global_xper.to_csv(os.path.join(exp_path, 'xper_values', f"cluster_{cluster}_global_xper_test.csv"), index=True)



    # Compute SHAP values.
    explainer = shap.Explainer(cluster_model, seed=42)
    shap_values_train = explainer(X_train)
    data['shap_train_values'] = shap_values_train

    shap_values_test = explainer(X_test)
    data['shap_test_values'] = shap_values_test

    # Generate Permutation Importance values.
    perm_importance_train = permutation_importance(cluster_model, X_train, y_train,
                                           n_repeats=30, random_state=42, scoring='roc_auc')
    data['perm_importance_train'] = perm_importance_train

    perm_importance_test = permutation_importance(cluster_model, X_test, y_test,
                                           n_repeats=30, random_state=42, scoring='roc_auc')
    data['perm_importance_test'] = perm_importance_test

    return data

# =============================================================================
# Function: descriptive_stats_and_plots
# =============================================================================
def descriptive_stats_and_plots(df, title_prefix, output_dir):
    """
    Generate descriptive statistics and plots (histogram, boxplot, KDE) for the given DataFrame.
    
    Parameters:
      df (pd.DataFrame): The DataFrame containing XPER values.
      title_prefix (str): A prefix to use in plot titles and filenames.
      output_dir (str): Directory where plots and CSVs will be saved.
      
    Returns:
      pd.DataFrame: The descriptive statistics of the DataFrame.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Compute descriptive statistics.
    desc_stats = df.describe()

    # Save descriptive statistics as CSV.
    stats_file = os.path.join(output_dir, f"{title_prefix}_descriptive_stats.csv")
    desc_stats.to_csv(stats_file)

    # Plot histograms for each column.
    for col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), kde=False, bins=30)
        plt.title(f"{title_prefix} - Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        hist_file = os.path.join(output_dir, f"{title_prefix}_histogram_{col}.png")
        plt.savefig(hist_file, dpi=300, bbox_inches='tight')
        plt.close()

    # Plot boxplots for each column.
    for col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col].dropna())
        plt.title(f"{title_prefix} - Boxplot of {col}")
        plt.xlabel(col)
        box_file = os.path.join(output_dir, f"{title_prefix}_boxplot_{col}.png")
        plt.savefig(box_file, dpi=300, bbox_inches='tight')
        plt.close()

    # Plot kernel density estimates for each column.
    for col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.kdeplot(df[col].dropna(), fill=True, warn_singular=False)
        plt.title(f"{title_prefix} - KDE of {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        kde_file = os.path.join(output_dir, f"{title_prefix}_kde_{col}.png")
        plt.savefig(kde_file, dpi=300, bbox_inches='tight')
        plt.close()

    return desc_stats

# =============================================================================
# Function: compare_experiments
# =============================================================================
def compare_experiments(exp_data_list, metric_name, output_dir):
    """
    Compare a specified metric (e.g., global XPER values) across multiple experiments.
    
    Parameters:
      exp_data_list (list of tuples): List of (experiment_name, data_dict) for each experiment.
      metric_name (str): The key in the data_dict to compare (e.g., 'global_train_xper').
      output_dir (str): Directory where the comparison plots will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combined_df = pd.DataFrame()
    for exp_name, data in exp_data_list:
        df = data.get(metric_name)
        if df is not None:
            df = df.copy()
            df['experiment'] = exp_name
            combined_df = pd.concat([combined_df, df], axis=0)
        else:
            logger.info(f"No {metric_name} data for experiment {exp_name}")

    if combined_df.empty:
        logger.warning("No combined data available for comparison.")
        return

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_combined_df = combined_df.select_dtypes(include=numerics)

    for metric_col in numeric_combined_df.columns:
        # Plot histogram stratified by experiment.
        plt.figure(figsize=(10, 8))
        sns.histplot(data=combined_df, x=metric_col, hue='experiment', bins=30, kde=False)
        plt.title(f"Histogram of {metric_name} across experiments")
        hist_cmp_file = os.path.join(output_dir, f"{metric_name}_comparison_histogram.png")
        plt.savefig(hist_cmp_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot boxplot.
        plt.figure(figsize=(10, 8))
        sns.boxplot(data=combined_df, x='experiment', y=metric_col)
        plt.title(f"Boxplot of {metric_name} across experiments")
        box_cmp_file = os.path.join(output_dir, f"{metric_name}_comparison_boxplot.png")
        plt.savefig(box_cmp_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot KDE.
        plt.figure(figsize=(10, 8))
        sns.kdeplot(data=combined_df, x=metric_col, hue='experiment', fill=True)
        plt.title(f"KDE of {metric_name} across experiments")
        kde_cmp_file = os.path.join(output_dir, f"{metric_name}_comparison_kde.png")
        plt.savefig(kde_cmp_file, dpi=300, bbox_inches='tight')
        plt.close()

# =============================================================================
# Function: store_shap_plots
# =============================================================================
def store_shap_plots(data, output_dir, experiment_name):
    """
    Generate and save SHAP summary plots for the training and test sets
    in a dedicated subfolder.
    """
    shap_train = data.get('shap_train_values', None)
    shap_test = data.get('shap_test_values', None)
    X_train = data.get('X_train', None)
    X_test = data.get('X_test', None)
    
    if shap_train is not None and X_train is not None:
        shap_dir = os.path.join(output_dir, 'shap_plots')
        if not os.path.exists(shap_dir):
            os.makedirs(shap_dir)
        plt.figure()
        shap.summary_plot(shap_train, features=X_train, show=False)
        train_plot_file = os.path.join(shap_dir, f"{experiment_name}_shap_summary_train.png")
        plt.savefig(train_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    if shap_test is not None and X_test is not None:
        shap_dir = os.path.join(output_dir, 'shap_plots')
        if not os.path.exists(shap_dir):
            os.makedirs(shap_dir)
        plt.figure()
        shap.summary_plot(shap_test, features=X_test, show=False)
        test_plot_file = os.path.join(shap_dir, f"{experiment_name}_shap_summary_test.png")
        plt.savefig(test_plot_file, dpi=300, bbox_inches='tight')
        plt.close()

# =============================================================================
# Function: compare_global_importance_metrics
# =============================================================================
def compare_global_importance_metrics(exp_data_list, output_dir):
    """
    Compare global values for Permutation Importance (PI), SHAP, and XPER
    across experiments using a grouped bar chart.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for exp_name, data in exp_data_list:
        # Global XPER (assumed to be a single-row DataFrame with features as columns)
        X_train = data.get('X_train', None)
        global_xper_df = data.get('global_train_xper', None)
        if global_xper_df is None:
            continue
        else:
            xper_series = pd.Series(global_xper_df.iloc[1:]['Global XPER'].values, index=X_train.columns, name='XPER').reset_index().rename(columns={'index': 'Feature'})


        # Permutation Importance from training set.
        perm_imp = data.get('perm_importance_train', None)
        if perm_imp is not None and X_train is not None:
            pi_series = pd.Series(perm_imp.importances_mean, index=X_train.columns, name='PI').reset_index().rename(columns={'index': 'Feature'})
        else:
            pi_series = pd.DataFrame(columns=['Feature', 'PI'])
        
        # Global SHAP importance (mean absolute value).
        shap_train = data.get('shap_train_values', None)
        if shap_train is not None:
            shap_importance = np.abs(shap_train.values).mean(axis=0)
            shap_series = pd.Series(shap_importance, index=shap_train.feature_names, name='SHAP').reset_index().rename(columns={'index': 'Feature'})
        else:
            shap_series = pd.DataFrame(columns=['Feature', 'SHAP'])
        
        # Merge the three metrics.
        merged = xper_series.merge(pi_series, on='Feature', how='outer').merge(shap_series, on='Feature', how='outer')
        merged.fillna(0, inplace=True)

        for col in ['XPER', 'PI', 'SHAP']:
            total = merged[col].sum()
            if total != 0:
                merged[col] = merged[col] / total
                
        # Define the desired order of features
        desired_order = ["Age", "Car price", "Job tenure", "Funding amount", "Down payment", "Loan duration"]

        # Filter the DataFrame to include only the features from the desired order that exist in the data
        features_to_plot = [feat for feat in desired_order if feat in merged["Feature"].values]
        merged = merged[merged["Feature"].isin(features_to_plot)].copy()

        # Set the 'Feature' column as an ordered categorical so that it follows the desired order
        merged["Feature"] = pd.Categorical(merged["Feature"], categories=features_to_plot, ordered=True)
        merged.sort_values("Feature", inplace=True)

        # Create the bar plot with different shades of red for each importance metric
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.25
        ind = np.arange(len(merged))

        ax.bar(ind - bar_width, merged['XPER'], width=bar_width, label='XPER')
        ax.bar(ind, merged['PI'], width=bar_width, label='Permutation Importance')
        ax.bar(ind + bar_width, merged['SHAP'], width=bar_width, label='SHAP')

        ax.set_xticks(ind)
        ax.set_xticklabels(merged['Feature'], rotation=45, ha='right')
        ax.set_ylabel("Contribution (%)")
        ax.set_title("Global Importance Comparison")
        ax.legend()

        cmp_file = os.path.join(output_dir, f"{exp_name}_global_importance_comparison.png")
        plt.savefig(cmp_file, dpi=300, bbox_inches='tight')
        plt.close()
# =============================================================================
# Function: compare_local_shap_xper
# =============================================================================
def compare_local_shap_xper(data, output_dir, experiment_name):
    """
    Compare local (per-instance) distributions for SHAP vs XPER for each feature using box plots.
    """
    individual_df = data.get('individual_train_xper', None)
    shap_train = data.get('shap_train_values', None)
    if individual_df is None or shap_train is None:
        return
    
    # Consider only numeric columns as candidate features.
    features = [col for col in individual_df.columns if np.issubdtype(individual_df[col].dtype, np.number)]
    shap_df = pd.DataFrame(shap_train.values, columns=shap_train.feature_names)
    
    local_cmp_dir = os.path.join(output_dir, 'local_comparison')
    if not os.path.exists(local_cmp_dir):
        os.makedirs(local_cmp_dir)
    for feature in features:
        if feature not in shap_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8,6))
        box_data = pd.DataFrame({
            'XPER': individual_df[feature].dropna(),
            'SHAP': shap_df[feature].dropna()
        })
        box_data_melted = box_data.melt(var_name='Metric', value_name='Value')
        sns.boxplot(x='Metric', y='Value', data=box_data_melted, ax=ax)
        ax.set_title(f"Local Comparison for {feature}")
        cmp_file = os.path.join(local_cmp_dir, f"{experiment_name}_local_comparison_{feature}.png")
        plt.savefig(cmp_file, dpi=300, bbox_inches='tight')
        plt.close()

# =============================================================================
# Function: compare_xper_feature_distribution
# =============================================================================
def compare_xper_feature_distribution(data, output_dir, experiment_name):
    """
    Compare the distribution of XPER values to the actual feature value distribution.
    For each feature, creates one chart with two subplots (train and test) using normalized values.
    """
    individual_train_df = data.get('individual_train_xper', None)
    individual_test_df = data.get('individual_test_xper', None)
    X_train = data.get('X_train', None)
    X_test = data.get('X_test', None)
    if individual_train_df is None or individual_test_df is None or X_train is None or X_test is None:
        logger.info(f"Insufficient data for XPER vs feature distribution comparison in experiment {experiment_name}")
        return
    
    # Only consider numeric features from individual XPER data.
    features = [col for col in individual_train_df.columns if np.issubdtype(individual_train_df[col].dtype, np.number)]
    
    dist_cmp_dir = os.path.join(output_dir, 'feature_value_comparison')
    if not os.path.exists(dist_cmp_dir):
        os.makedirs(dist_cmp_dir)
        
    def normalize_series(series):
        """Min-max normalize a Pandas Series to [0,1]."""
        if series.max() - series.min() != 0:
            return (series - series.min()) / (series.max() - series.min())
        else:
            return series
    
    for feature in features:
        if feature not in X_train.columns or feature not in X_test.columns:
            continue
        
        # Normalize the series for training set.
        xper_train = individual_train_df[feature].dropna()
        feature_train = X_train[feature].dropna()
        xper_train_norm = normalize_series(xper_train)
        feature_train_norm = normalize_series(feature_train)
        
        # Normalize the series for test set.
        xper_test = individual_test_df[feature].dropna()
        feature_test = X_test[feature].dropna()
        xper_test_norm = normalize_series(xper_test)
        feature_test_norm = normalize_series(feature_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(14,6))
        
        # Training set subplot.
        sns.kdeplot(xper_train_norm, label='XPER (normalized)', ax=axes[0])
        sns.kdeplot(feature_train_norm, label='Feature Value (normalized)', ax=axes[0])
        axes[0].set_title(f"Train: {feature}")
        axes[0].legend()
        
        # Test set subplot.
        sns.kdeplot(xper_test_norm, label='XPER (normalized)', ax=axes[1])
        sns.kdeplot(feature_test_norm, label='Feature Value (normalized)', ax=axes[1])
        axes[1].set_title(f"Test: {feature}")
        axes[1].legend()
        cmp_file = os.path.join(dist_cmp_dir, f"{experiment_name}_xper_vs_feature_{feature}.png")
        plt.savefig(cmp_file, dpi=300, bbox_inches='tight')
        plt.close()

# =============================================================================
# Function: stratified_analysis
# =============================================================================
def stratified_analysis(individual_xper_df, strat_col, output_dir, experiment_name):
    """
    Perform stratified analysis on individual XPER values based on a stratification column.
    
    Parameters:
      individual_xper_df (pd.DataFrame): DataFrame of individual XPER values.
      strat_col (str): Column name to use for stratification (e.g., "Cluster").
      output_dir (str): Directory where the stratified plots will be saved.
      experiment_name (str): Name of the experiment (for labeling purposes).
    """
    if strat_col not in individual_xper_df.columns:
        logger.warning(f"Stratification column '{strat_col}' not found in experiment {experiment_name}.")
        return

    strata = individual_xper_df[strat_col].unique()
    for stratum in strata:
        subset = individual_xper_df[individual_xper_df[strat_col] == stratum]
        title_prefix = f"{strat_col} {stratum} XPER"
        descriptive_stats_and_plots(subset, title_prefix, output_dir)

# =============================================================================
# Function: main
# =============================================================================
def main():

    subfolders = [
        item for item in os.listdir(BASE_EXPERIMENT_DIR)
        if os.path.isdir(os.path.join(BASE_EXPERIMENT_DIR, item))
    ]

    # Filter for specific experiment folders.
    exp_paths = [path for path in subfolders if path in ['experiment_results_17022025125138', 'experiment_results_17022025212918', 'experiment_results_17022025231258', 'experiment_results_18022025113359', 'experiment_results_18022025120717']]
    exp_data_list = [] 
    for exp_path in exp_paths:
        for cluster in [0, 1]:
            exp_name = os.path.basename(exp_path) + f"_Cluster{cluster}"
            data = load_experiment_data(os.path.join(BASE_EXPERIMENT_DIR, exp_path, 'credit_risk'), cluster)
            exp_data_list.append((exp_name, data))

    # Create a main output directory for all analysis plots.
    output_dir = os.path.join(BASE_EXPERIMENT_DIR, "analysis_plots_clusters")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # For each experiment, analyze individual and global XPER values and perform additional comparisons.
    for exp_name, data in exp_data_list:
        exp_output_dir = os.path.join(output_dir, exp_name)
        
        if not os.path.exists(exp_output_dir):
            os.makedirs(exp_output_dir)

        # Analyze individual XPER values.
        individual_df = data.get('individual_train_xper')
        if individual_df is not None:
            descriptive_stats_and_plots(individual_df, "Train_XPER", exp_output_dir)
            if 'Cluster' in individual_df.columns:
                stratified_analysis(individual_df, 'Train Cluster', exp_output_dir, exp_name)
        else:
            logger.info(f"No individual XPER data for experiment {exp_name}")

        individual_df_test = data.get('individual_test_xper')
        if individual_df_test is not None:
            descriptive_stats_and_plots(individual_df_test, "Test_XPER", exp_output_dir)
            if 'Cluster' in individual_df_test.columns:
                stratified_analysis(individual_df_test, 'Test Cluster', exp_output_dir, exp_name)
        else:
            logger.info(f"No individual XPER test data for experiment {exp_name}")

        # Analyze global XPER values.
        global_df = data.get('global_train_xper')
        if global_df is not None:
            descriptive_stats_and_plots(global_df, "Global_XPER_Train", exp_output_dir)
        else:
            logger.info(f"No global XPER train data for experiment {exp_name}")

        global_df_test = data.get('global_test_xper')
        if global_df_test is not None:
            descriptive_stats_and_plots(global_df_test, "Global_XPER_Test", exp_output_dir)
        else:
            logger.info(f"No global XPER test data for experiment {exp_name}")

        # Save SHAP summary plots.
        store_shap_plots(data, exp_output_dir, exp_name)

        # Compare local SHAP vs XPER distributions.
        compare_local_shap_xper(data, exp_output_dir, exp_name)

        # Compare XPER distributions with actual feature value distributions.
        compare_xper_feature_distribution(data, exp_output_dir, exp_name)

    # Compare global importance metrics (XPER, PI, SHAP) across experiments.
    global_cmp_dir = os.path.join(output_dir, "comparison")
    compare_global_importance_metrics(exp_data_list, global_cmp_dir)

    # Also compare global XPER values across experiments.
    compare_experiments(exp_data_list, 'global_train_xper', global_cmp_dir)
    compare_experiments(exp_data_list, 'global_test_xper', global_cmp_dir)

# =============================================================================
# Entry point
# =============================================================================
if __name__ == '__main__':
    main()
