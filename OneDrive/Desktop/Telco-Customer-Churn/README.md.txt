 📊 Customer Churn Prediction using Machine Learning

This project focuses on predicting customer churn for a telecom company using machine learning techniques.  
Customer churn is a critical business problem where identifying customers likely to leave helps companies take proactive retention actions.



 🚀 Project Overview

- Built an end-to-end machine learning pipeline for churn prediction
- Performed data cleaning, feature engineering, and encoding
- Addressed class imbalance and optimized the model for business impact
- Evaluated the model using appropriate metrics beyond accuracy



 🧠 Problem Statement

Customer churn occurs when customers stop using a company’s services.  
The objective of this project is to **predict whether a customer will churn (Yes/No)** based on their service usage, contract details, and billing information.



 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook



 🔍 Data Preprocessing & Feature Engineering

- Handled missing values in 'TotalCharge'
- Applied log transformation to reduce skewness
- Capped outliers in 'MonthlyCharges' using IQR method
- Encoded binary categorical variables using Label Encoding
- Applied One-Hot Encoding for multi-category features
- Dropped unnecessary columns like 'customerID'



 ⚙️ Model Used

- **RandomForestClassifier**
- Handled class imbalance using 'class_weight="balanced"'
- Optimized decision threshold to improve churn recall

---

 📈 Model Evaluation

 🔹 Confusion Matrix
Helps analyze false negatives and false positives, which are critical in churn prediction.

 🔹 ROC-AUC Score
- ROC-AUC: ~0.82
- Indicates strong separation between churn and non-churn customers

 🔹 Classification Metrics (Churn Class)
- Precision: ~0.58
- Recall: ~0.61
- F1-score: ~0.59

📌 Recall was prioritized since missing a churn customer is more costly than contacting a non-churn customer.

---

 📊 Visualizations Included

- Confusion Matrix Heatmap
- ROC Curve
- Precision–Recall Curve
- Feature Importance Plot

These visualizations help in understanding model performance and business impact.



 💼 Business Impact

- Reduced false negatives by tuning the probability threshold
- Identified key churn drivers such as contract type, tenure, and monthly charges
- Enables proactive customer retention strategies



  

