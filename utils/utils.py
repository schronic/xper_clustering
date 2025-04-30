import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV


def evaluate_model(model, X_train, X_test, y_train, y_test, model_type):
    """Evaluate model performance based on problem type."""
    if model_type == 'binary':
        return roc_auc_score(y_train, model.predict(X_train)), roc_auc_score(y_test, model.predict(X_test))
    elif model_type == 'multiclass':
        return roc_auc_score(y_train, model.predict_proba(X_train), multi_class="ovr"), roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
    elif model_type == 'regression':
        return r2_score(y_train, model.predict(X_train)), r2_score(y_test, model.predict(X_test))


def initiate_model(model_type: str, num_classes: int = None, X_train=None, y_train=None):
    """
    Initialize an XGBoost model for overfitting prevention and tune it using RandomizedSearchCV.
    """
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train must be provided for random search tuning.")
    
    # Define base parameters and parameter distributions for tuning.
    if model_type == "regression":
        base_params = {"objective": "reg:squarederror", "seed": 42, "n_estimators": 100}
        ModelClass = xgb.XGBRegressor
        scoring = "neg_mean_squared_error"
    elif model_type == "binary":
        base_params = {"objective": "binary:logistic", "seed": 42, "n_estimators": 100}
        ModelClass = xgb.XGBClassifier
        scoring = "roc_auc"
    elif model_type == "multiclass":
        if num_classes is None:
            raise ValueError("num_classes must be provided for multiclass classification.")
        base_params = {"objective": "multi:softprob", "num_class": num_classes, "seed": 42, "n_estimators": 100}
        ModelClass = xgb.XGBClassifier
        scoring = "roc_auc_ovr"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Common parameter distributions.
    param_distributions = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "alpha": [0, 0.5, 1],
        "lambda": [0, 0.5, 1]
    }

    # Initialize the model with base parameters.
    model = ModelClass(**base_params)
    
    # Set up RandomizedSearchCV.
    random_search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=20,
        scoring=scoring,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    return best_model



def identify_problem_type(dataset_name: str, y_train: pd.Series, y_test: pd.Series, target_col: str, label_encoder: LabelEncoder):
    """Determine if the dataset is for classification (binary/multiclass) or regression."""
    classification, num_classes = False, None
    
    if dataset_name in ["Loan Status", "Bank Marketing", "Credit Risk"]:
        model_type = "binary"
        classification = True
    elif dataset_name in ["Bike Sharing", "Abalone", "Boston Housing"]:
        model_type = "regression"
    elif dataset_name in ["Wine Quality", "Iris"]:
        model_type = "multiclass"
        classification = True
        num_classes = len(label_encoder.classes_)
    
    if classification:
        y_train = pd.DataFrame(label_encoder.fit_transform(y_train), columns=[target_col])
        y_test = pd.DataFrame(label_encoder.transform(y_test), columns=[target_col])
    
    return model_type, num_classes #TODO: completely rip out the classification tab


def highlight_best_train_test(df):
    """
    Returns a DataFrame of CSS styles highlighting:
      - max values in columns_to_max for each of train vs test rows
      - min values in columns_to_min for each of train vs test rows
    """

    # We'll store styles in a DataFrame of the same shape/index/columns.
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    # Boolean masks for train/test rows
    train_mask = df["Data"].str.contains("train")
    test_mask  = df["Data"].str.contains("test")

    # Decide which columns to maximize and which to minimize
    columns_to_max = [
        "AUC Score", "Accuracy", "Precision", "Recall (TPR)",
        "F1 Score", "True Negative Rate (TNR)"
    ]
    columns_to_min = [
        "False Positive Rate (FPR)", "False Negative Rate (FNR)"
    ]

    # Highlight maxima in columns_to_max
    for col in columns_to_max:
        # Train
        train_vals = df.loc[train_mask, col]
        if not train_vals.empty:
            max_idx = train_vals.idxmax()
            styles.loc[max_idx, col] = "background-color: #A6F4A6;" 

        # Test
        test_vals = df.loc[test_mask, col]
        if not test_vals.empty:
            max_idx = test_vals.idxmax()
            styles.loc[max_idx, col] = "background-color: #F7CA9F;"  

    # Highlight minima in columns_to_min
    for col in columns_to_min:
        # Train
        train_vals = df.loc[train_mask, col]
        if not train_vals.empty:
            min_idx = train_vals.idxmin()
            styles.loc[min_idx, col] = "background-color: #A6F4A6;"

        # Test
        test_vals = df.loc[test_mask, col]
        if not test_vals.empty:
            min_idx = test_vals.idxmin()
            styles.loc[min_idx, col] = "background-color: #F7CA9F;"

    return styles

def weighted_aggregator(group):
    """Compute weighted averages over all numeric columns (except Cluster Size),
       weighting by 'Cluster Size'. Also sum the 'Cluster Size' itself."""

    numeric_cols = group.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove("Cluster ID")
    
    if "Cluster Size" in numeric_cols:
        numeric_cols.remove("Cluster Size")

    total_size = group["Cluster Size"].sum()
    result = {}

    for col in numeric_cols:
        result[col] = (group[col] * group["Cluster Size"]).sum() / total_size

    result["Cluster Size"] = total_size
    return pd.Series(result)