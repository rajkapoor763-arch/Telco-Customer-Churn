import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt

model_path = "model.pkl"
scaler_path = "scaler.pkl"
data_path = "67f168f6afc1b.csv"
output_path= "telco_churn_predictions.csv"


model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("Model and scaler loaded successfully.")

df=pd.read_csv(data_path)
print(df.shape)
print(df.head())

customer_ids=df["customerID"]

df=df.drop("customerID",axis=1)

df["Churn"]=df["Churn"].map({"Yes": 1,"No": 0})

print(df.head())


categorical_cols=df.select_dtypes(include=["object"]).columns
df_encoded=pd.get_dummies(df,columns=categorical_cols,drop_first=True)
print(f"processed data shape{df_encoded.shape}")
print(df_encoded.head())

feature_columns = [
    'tenure', 'MonthlyCharges', 'TotalCharges_log', 'SeniorCitizen',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]


for col in feature_columns:
    if col not in df_encoded.columns:
        df_encoded[col]=0


df_encoded=df_encoded[feature_columns]

x_scaled=scaler.transform(df_encoded)

y_prob=model.predict_proba(x_scaled)[:,1]
df["Churn_Probability"]=y_prob
df["Predicted_Churn"]=(y_prob >=0.5).astype(int)

df["customerID"]=customer_ids
df.to_csv_path,index=False
print(f"predictions saved to '{output_path}'")