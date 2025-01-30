
from sklearn.metrics import roc_auc_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def evaluate_model(model, X_train, X_test, y_train, y_test, model_type):
    """Evaluate model performance based on problem type."""
    if model_type == 'binary':
        return roc_auc_score(y_train, model.predict(X_train)), roc_auc_score(y_test, model.predict(X_test))
    elif model_type == 'multiclass':
        return roc_auc_score(y_train, model.predict_proba(X_train), multi_class="ovr"), roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
    elif model_type == 'regression':
        return r2_score(y_train, model.predict(X_train)), r2_score(y_test, model.predict(X_test))

def initiate_model(model_type: str, num_classes: int = None):
    """Initialize an XGBoost model based on the problem type."""
    if model_type == "binary":
        return XGBClassifier(eval_metric="error", use_label_encoder=False)
    elif model_type == "regression":
        return XGBRegressor()
    elif model_type == "multiclass":
        return XGBClassifier(objective="multi:softmax", num_class=num_classes)
    

def identify_problem_type(dataset_name: str, y_train: pd.Series, y_test: pd.Series, target_col: str, label_encoder: LabelEncoder):
    """Determine if the dataset is for classification (binary/multiclass) or regression."""
    classification, num_classes = False, None
    
    if dataset_name in ["Loan Status", "Bank Marketing"]:
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
    
    return model_type, classification, num_classes


