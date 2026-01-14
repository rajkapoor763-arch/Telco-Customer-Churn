import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Paths
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
# Load data
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
# Align features using model input size
# -----------------------------
expected_features = model.n_features_in_

if df_encoded.shape[1] > expected_features:
    df_encoded = df_encoded.iloc[:, :expected_features]
elif df_encoded.shape[1] < expected_features:
    for i in range(expected_features - df_encoded.shape[1]):
        df_encoded[f"missing_{i}"] = 0

# -----------------------------
# 🚀 SCALE FULL DATA (IMPORTANT FIX)
# -----------------------------
df_encoded_scaled = scaler.transform(df_encoded.values)

# -----------------------------
# Predict
# -----------------------------
y_prob = model.predict_proba(df_encoded_scaled)[:, 1]

df['Churn_Probability'] = y_prob
df['Predicted_Churn'] = (y_prob >= 0.3).astype(int)
df['customerID'] = customer_ids

# -----------------------------
# Save output
# -----------------------------
df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to '{output_path}'")

# -----------------------------
# Plot
# -----------------------------
plt.hist(df['Churn_Probability'], bins=20)
plt.title("Churn Probability Distribution")
plt.xlabel("Churn Probability")
plt.ylabel("Customers")
plt.show()


