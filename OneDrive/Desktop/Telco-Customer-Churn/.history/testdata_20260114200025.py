import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# File paths
# -----------------------------
model_path = "model.pkl"
scaler_path = "scaler.pkl"
data_path = "67f168f6afc1b.csv"
output_path = "telco_churn_predictions.csv"

# -----------------------------
# Load model & scaler
# -----------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("✅ Model and scaler loaded successfully.")

# -----------------------------
# Load test data
# -----------------------------
df = pd.read_csv(data_path)
print(f"Data shape: {df.shape}")

customer_ids = df["customerID"]

# -----------------------------
# Preprocessing
# -----------------------------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['TotalCharges_log'] = np.log1p(df['TotalCharges'])

df_model = df.drop("customerID", axis=1)

if 'Churn' in df_model.columns:
    df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})

# -----------------------------
# One-hot encoding
# -----------------------------
categorical_cols = df_model.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=False)

# -----------------------------
# SAFE feature alignment (NO feature.pkl)
# -----------------------------
# Use model expected input size
expected_features = model.n_features_in_

current_features = df_encoded.shape[1]

if current_features > expected_features:
    df_encoded = df_encoded.iloc[:, :expected_features]
elif current_features < expected_features:
    for i in range(expected_features - current_features):
        df_encoded[f"missing_{i}"] = 0

# -----------------------------
# Scale numeric columns
# -----------------------------
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges_log']
df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

# -----------------------------
# Predict churn
# -----------------------------
y_prob = model.predict_proba(df_encoded)[:, 1]

df['Churn_Probability'] = y_prob
df['Predicted_Churn'] = (y_prob >= 0.5).astype(int)
df['customerID'] = customer_ids

# -----------------------------
# Save predictions
# -----------------------------
df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to '{output_path}'")

# -----------------------------
# Plot
# -----------------------------
plt.hist(df['Churn_Probability'], bins=20)
plt.title("Churn Probability Distribution")
plt.xlabel("Churn Probability")
plt.ylabel("Number of Customers")
plt.show()

