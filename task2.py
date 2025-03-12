import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- 1. Dataset Generation (Simplified Simulation) ---
# For a real-world scenario, you'd use a dataset like the one from Kaggle:
# https://www.kaggle.com/mlg-ulb/creditcardfraud
# or generate your own with more realistic features.
# Here, we generate a highly simplified synthetic dataset.

np.random.seed(42)  # For reproducibility

n_samples = 10000
n_fraud = 100  # 1% fraud rate

# Generate legitimate transactions
data_legit = pd.DataFrame({
    'amount': np.random.normal(100, 50, n_samples),
    'feature1': np.random.rand(n_samples),
    'feature2': np.random.rand(n_samples),
    'feature3': np.random.rand(n_samples),
    'class': 0  # 0 for legitimate
})

# Generate fraudulent transactions (with different characteristics)
data_fraud = pd.DataFrame({
    'amount': np.random.normal(500, 200, n_fraud), #higher amounts
    'feature1': np.random.rand(n_fraud) + 0.5, #different feature distributions
    'feature2': np.random.rand(n_fraud) + 0.5,
    'feature3': np.random.rand(n_fraud) + 0.5,
    'class': 1  # 1 for fraud
})

# Combine and shuffle the data
data = pd.concat([data_legit, data_fraud], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# --- 2. Data Preprocessing ---

X = data.drop('class', axis=1)
y = data['class']

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Model Training and Evaluation ---

# Logistic Regression
logreg = LogisticRegression(random_state=42, solver='liblinear') #liblinear handles small datasets well
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

print("Logistic Regression:")
print(confusion_matrix(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\nDecision Tree:")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# --- 4. Handling Imbalanced Data (Important for Fraud Detection) ---
# Because fraud is rare, the dataset is highly imbalanced. Techniques like
# oversampling (SMOTE), undersampling, or using algorithms designed for
# imbalanced data (e.g., class_weight='balanced' in some scikit-learn models)
# are crucial in real-world scenarios.

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

rf_balanced = RandomForestClassifier(random_state=42)
rf_balanced.fit(X_resampled, y_resampled)
y_pred_rf_balanced = rf_balanced.predict(X_test)

print("\nRandom Forest (SMOTE Resampled):")
print(confusion_matrix(y_test, y_pred_rf_balanced))
print(classification_report(y_test, y_pred_rf_balanced))
accuracy=accuracy_score(y_test, y_pred_rf_balanced)
print("Accuracy:",accuracy)
# --- 5. Saving the Dataset ---
data.to_csv("credit_card_data.csv", index=False) #save the generated dataset. 