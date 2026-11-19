import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "login_activity.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
PLOTS_DIR = ROOT / "outputs" / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

TARGET = "is_suspicious"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# --------------------------------------------------
# FEATURES
# --------------------------------------------------
numeric_features = [
    "login_hour",
    "failed_attempts",
    "ip_risk_score",
    "unusual_location",
    "device_trusted",
    "account_age_days",
    "login_success",
]

categorical_features = ["region_group"]

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
numeric_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = Pipeline(
    [("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))]
)

print("=" * 60)
print("LOGISTIC REGRESSION - THREAT CLASSIFICATION")
print("=" * 60)

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nModel Performance:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"Training time: {training_time:.4f} seconds")

print("\nClassification Report:")
report = classification_report(
    y_test, y_pred, target_names=["Normal", "Suspicious"], zero_division=0
)
print(report)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Suspicious"],
    yticklabels=["Normal", "Suspicious"],
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig(PLOTS_DIR / "07_logistic_confusion_matrix.png", dpi=300)
plt.close()

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
metrics = {
    "model": "Logistic Regression",
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "training_time_seconds": float(training_time),
}

with open(RESULTS_DIR / "logistic_regression_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open(RESULTS_DIR / "logistic_regression_report.txt", "w") as f:
    f.write(report)

joblib.dump(model, RESULTS_DIR / "logistic_regression_model.joblib")

# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------
print("\nConfusion Matrix:")
print(cm)

print("\nSaved outputs:")
print("- Logistic Regression confusion matrix plot")
print("- Metrics JSON")
print("- Classification report")
print("- Trained model")

print("\nML training completed successfully!")
