import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_datasets(data_list):

    datasets = {}

    ### **1. Load Iris Dataset**
    if "Iris" in data_list:
        from XPER.datasets.load_data import iris
        iris_df = pd.DataFrame(iris().data, columns=iris().feature_names)
        iris_df["species"] = iris().target
        datasets["Iris"] = (iris_df, "species")

    ### **2. Load Loan Status Dataset**
    if "Loan Status" in data_list:
        from XPER.datasets.load_data import loan_status
        loan = loan_status()
        datasets["Loan Status"] = (loan, "Loan_Status")

    ### **3. Load Boston Housing Dataset**
    if "Boston Housing" in data_list:
        from XPER.datasets.load_data import boston
        boston_data = boston()
        datasets["Boston Housing"] = (boston_data, "medv")

    ### **4. Load Wine Quality Dataset**
    if "Wine Quality" in data_list:
        wine_quality_dir = os.path.join("data", "wine+quality")
        red_wine_file = os.path.join(wine_quality_dir, "winequality-red.csv")
        white_wine_file = os.path.join(wine_quality_dir, "winequality-white.csv")

        if os.path.exists(red_wine_file) and os.path.exists(white_wine_file):
            red_wine = pd.read_csv(red_wine_file, sep=";")
            white_wine = pd.read_csv(white_wine_file, sep=";")
            wine = pd.concat([red_wine, white_wine], ignore_index=True)
            datasets["Wine Quality"] = (wine, "quality")

    ### **5. Load Abalone Dataset**
    if "Abalone" in data_list:
        abalone_dir = "abalone"
        abalone_file = os.path.join(abalone_dir, "abalone.data")

        if os.path.exists(abalone_file):
            abalone_columns = ["Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight",
                            "VisceraWeight", "ShellWeight", "rings"]
            abalone = pd.read_csv(abalone_file, header=None, names=abalone_columns)
            datasets["Abalone"] = (abalone, "rings")

    ### **6. Load Bank Marketing Dataset**
    if "Bank Marketing" in data_list:
        bank_marketing_dir = os.path.join("data", "bank+marketing", "bank")
        bank_file = os.path.join(bank_marketing_dir, "bank.csv")

        if os.path.exists(bank_file):
            bank = pd.read_csv(bank_file, sep=";")

            label_encoder = LabelEncoder()
            bank['y'] = label_encoder.fit_transform(bank['y'])

            datasets["Bank Marketing"] = (bank, "y")

    ### **7. Load Bike Sharing Dataset**
    if "Bike Sharing" in data_list:
        bike_sharing_dir = os.path.join("data", "bike+sharing+dataset")
        hourly_file = os.path.join(bike_sharing_dir, "hour.csv")

        if os.path.exists(hourly_file):
            bike = pd.read_csv(hourly_file)
            datasets["Bike Sharing"] = (bike, "cnt")  # Adjust target column if needed


    ### **5. Load credit_risk Dataset**
    if "Credit Risk" in data_list:
        credit_risk_dir = os.path.join("data", "credit_risk")
        credit_risk_file = os.path.join(credit_risk_dir, "dataproject2024.xlsx")

        if os.path.exists(credit_risk_file):
            credit_risk = pd.read_excel(credit_risk_file, usecols=["Job tenure", "Age", 
                                                                "Car price", "Funding amount", "Down payment", 
                                                                "Loan duration", "Monthly payment", "Credit event", 
                                                                "Married", "Homeowner", "Default (y)"])
            datasets["Credit Risk"] = (credit_risk, "Default (y)")

    return datasets

