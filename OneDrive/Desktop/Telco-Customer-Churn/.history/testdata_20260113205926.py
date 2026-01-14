import pandas as pd
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



