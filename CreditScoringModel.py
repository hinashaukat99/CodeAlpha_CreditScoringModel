#  Introduction
#  𝗡𝗮𝗺𝗲: Hina
#  𝗦𝘁𝘂𝗱𝗲𝗻𝘁 𝗜𝗗: CA/SE3/13541
#  𝗗𝗼𝗺𝗮𝗶𝗻: Machine Learning
# =========================
# CREDIT SCORING MODEL
# =========================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


# =========================
# 2. LOAD DATASET
# =========================
df = pd.read_csv("german_credit_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# =========================
# 3. DATA CLEANING
# =========================

# Remove unnecessary column if exists
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Handle missing values
df = df.dropna()

# Convert categorical variables into numeric
df = pd.get_dummies(df, drop_first=True)

print("After preprocessing shape:", df.shape)

# Check columns (for understanding only)
print(df.columns)


# =========================
# 4. FEATURE ENGINEERING
# =========================

if 'Credit amount' in df.columns and 'Age' in df.columns:
    df['credit_per_age'] = df['Credit amount'] / df['Age']


# =========================
# 5. TARGET COLUMN 
# =========================

# Check column names (for reference)
print(df.columns)
target_column = "target_good"


# =========================
# 6. SPLIT FEATURES & TARGET
# =========================

X = df.drop(target_column, axis=1)
y = df[target_column]


# =========================
# 7. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 8. FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================
# 9. MODELS
# =========================

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}


# =========================
# 10. TRAIN & EVALUATE
# =========================

for name, model in models.items():
    print("\n========================")
    print("Model:", name)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ROC-AUC Score
    if hasattr(model, "predict_proba"):
        roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print("ROC-AUC Score:", roc)


# =========================
# DONE
# =========================
print("\n Credit Scoring Model Completed Successfully!")