import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt

model_path = "model.pkl"
scaler_path = "scaler.pkl"
data_path = "67f168f6afc1b.csv"


model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("Model and scaler loaded successfully.")

df=pd.read_csv(data_path)
print(df.shape)
print(df.head())

df=df.drop("customerID",axis=1)

df["Churn"]=df["Churn"].map({"Yes": 1,"No": 0})

print(df.head())

replace_cols=["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
    "StreamingTV","StreamingMovies","MultipleLines"
]

for col in replace_cols:
    df[col]=df[col].replace({"No internet service":"No","No phone service":"No"})

df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df["TotalCharges"]=df["TotalCharges"].fillna(0)
df["TotalCharges_log"]=np.log1p(df["TotalCharges"])
df=df.drop("TotalCharges",axis=1)

categorical_cols=df.select_dtypes(include=["object"]).columns
df_encoded=pd.get_dummies(df,columns=categorical_cols,drop_first=True)
print(f"processed data shape{df_encoded.shape}")
print(df_encoded.head())


feature_columns = [
    'tenure', 'MonthlyCharges', 'TotalCharges_log',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes',
    'PhoneService_Yes', 'MultipleLines_Yes', 'InternetService_Fiber optic',
    'InternetService_No', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
    'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes',
    'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'PaymentMethod_Bank transfer (automatic)'
]

for col in feature_columns:
    if col not in df_encoded.columns:
        df_encoded[col]=0


df_encoded=df_encoded[feature_columns]

x_scaled=scaler.transform(df_encoded)

y_prob=model.predict_proba(x_scaled)[:,1]
df["Churn_probability"]=y_prob
df["Predicted_Churn"]=(y_prob >=0.5).astype(int)

df.to_csv("telco_churn_predictions.csv",index=False)
print(f"✅ Predictions saved to '{"telco_churn_predictions.csv"}'")