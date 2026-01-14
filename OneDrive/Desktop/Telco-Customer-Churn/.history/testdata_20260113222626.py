import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# File paths
# -----------------------------
model_path = "model.pkl"
scaler_path = "scaler.pkl"
data_path = "67f168f6afc1b.csv"  # your CSV
output_path = "telco_churn_predictions.csv"

# -----------------------------
# Load model and scaler
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

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['TotalCharges_log'] = np.log1p(df['TotalCharges'])

# Drop customerID
df_model = df.drop("customerID", axis=1)

# Map target for reference (optional)
if 'Churn' in df_model.columns:
    df_model['Churn'] = df_model['Churn'].map({'Yes':1, 'No':0})

# -----------------------------
# One-hot encode categorical columns
# -----------------------------
categorical_cols = df_model.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=False)

# -----------------------------
# Align test data to training features
# -----------------------------
# Load feature columns used during training
# IMPORTANT: replace this list with the exact list from your training dataset
training_features = joblib.load("training_features.pkl")  # save this list when training!

# Add missing columns
for col in training_features:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Remove extra columns
df_encoded = df_encoded[training_features]

# -----------------------------
# Scale numeric columns
# -----------------------------
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges_log']
df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

# -----------------------------
# Predict churn
# -----------------------------
y_prob = model.predict_proba(df_encoded)[:,1]
df['Churn_Probability'] = y_prob
df['Predicted_Churn'] = (y_prob >= 0.5).astype(int)
df['customerID'] = customer_ids

# -----------------------------
# Save predictions
# -----------------------------
df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to '{output_path}'")

# Optional: plot
plt.hist(df['Churn_Probability'], bins=20, color='skyblue', edgecolor='black')
plt.title("Churn Probability Distribution")
plt.xlabel("Churn Probability")
plt.ylabel("Number of Customers")
plt.show()
