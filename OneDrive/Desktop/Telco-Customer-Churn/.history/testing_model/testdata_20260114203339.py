import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


model_path = "C:/Users/Dell/OneDrive/Desktop/Telco-Customer-Churn/model/model.pkl"
scaler_path = "C:/Users/Dell/OneDrive/Desktop/Telco-Customer-Churn/model/scaler.pkl"
data_path = "C:/Users/Dell/OneDrive/Desktop/Telco-Customer-Churn/testing_model"
output_path = "C:/Users/Dell/OneDrive/Desktop/Telco-Customer-Churn/testing_model/telco_churn_predictions.csv"


model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("✅ Model and scaler loaded successfully.")

df = pd.read_csv(data_path)
print(f"Data shape: {df.shape}")

customer_ids = df["customerID"]


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['TotalCharges_log'] = np.log1p(df['TotalCharges'])

df_model = df.drop("customerID", axis=1)

if 'Churn' in df_model.columns:
    df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})


categorical_cols = df_model.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=False)


expected_features = model.n_features_in_

if df_encoded.shape[1] > expected_features:
    df_encoded = df_encoded.iloc[:, :expected_features]
elif df_encoded.shape[1] < expected_features:
    for i in range(expected_features - df_encoded.shape[1]):
        df_encoded[f"missing_{i}"] = 0


df_encoded_scaled = scaler.transform(df_encoded.values)


y_prob = model.predict_proba(df_encoded_scaled)[:, 1]

df['Churn_Probability'] = y_prob
df['Predicted_Churn'] = (y_prob >= 0.3).astype(int)
df['customerID'] = customer_ids


df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to '{output_path}'")


sns.set_style("whitegrid")
plt.figure(figsize=(10,6))
sns.histplot(df['Churn_Probability'], bins=20, kde=True, color="#1f77b4", edgecolor='black')
plt.title("Customer Churn Probability Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Churn Probability", fontsize=14)
plt.ylabel("Number of Customers", fontsize=14)

plt.xlim(0,1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig("churn_probability_distribution.png", dpi=300)
plt.show()