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

categorical_cols=df.select_dtypes(include=["object"]).columns
df_encoded=pd.get_dummies(df,columns=categorical_cols,drop_first=True)

x=df_encoded.drop("Churn",axis=1)
y=df_encoded["Churn"]

x_scaled=scaler.transform(x)
y_prob=model.predict_proba(x_scaled)[:,1]

print(df[["Predicted_Churn","Churn_Probability"]].head())

df.to_csv("telco_churn_predictions.csv",index=False)
print("✅ Predictions saved to 'telco_churn_predictions.csv")





