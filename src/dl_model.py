
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "login_activity.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
PLOTS_DIR = ROOT / "outputs" / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["is_suspicious"])
y = df["is_suspicious"]

numeric_features = [
    "login_hour", "failed_attempts", "ip_risk_score",
    "unusual_location", "device_trusted", "account_age_days",
    "login_success"
]

categorical_features = ["region_group"]

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Train-test split (same split as Logistic Regression)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Deep Learning model: Multi-Layer Perceptron
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=300,
    early_stopping=True,
    random_state=42
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", mlp)
])

# Training
start = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start

# Predictions
y_pred = model.predict(X_test)

# Metrics
metrics = {
    "model": "MLP Neural Network",
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
    "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
    "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
    "training_time_seconds": round(training_time, 4)
}

print("=" * 60)
print("MLP NEURAL NETWORK - THREAT CLASSIFICATION")
print("=" * 60)
print("\nModel Performance:")
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall   : {metrics['recall']:.4f}")
print(f"F1-score : {metrics['f1_score']:.4f}")
print(f"Training time: {training_time:.4f} seconds")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Normal", "Suspicious"],
    yticklabels=["Normal", "Suspicious"]
)
plt.title("MLP Neural Network - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "08_mlp_confusion_matrix.png", dpi=300)
plt.close()

# Save results and model
with open(RESULTS_DIR / "mlp_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

joblib.dump(model, RESULTS_DIR / "mlp_model.joblib")

# Compare with Logistic Regression
with open(RESULTS_DIR / "logistic_regression_metrics.json") as f:
    lr_metrics = json.load(f)

comparison = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": lr_metrics["accuracy"],
        "Precision": lr_metrics["precision"],
        "Recall": lr_metrics["recall"],
        "F1 Score": lr_metrics["f1_score"],
        "Training Time (s)": lr_metrics["training_time_seconds"]
    },
    {
        "Model": "MLP Neural Network",
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1 Score": metrics["f1_score"],
        "Training Time (s)": metrics["training_time_seconds"]
    }
])

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv(RESULTS_DIR / "ml_dl_comparison.csv", index=False)

# Comparison visualization
plt.figure(figsize=(9, 5))
comparison.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
    kind="bar", figsize=(9, 5)
)
plt.title("Logistic Regression vs MLP - Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "09_model_comparison.png", dpi=300)
plt.close()

print("\nSaved outputs:")
print("- MLP confusion matrix")
print("- MLP metrics JSON")
print("- Trained MLP model")
print("- ML vs DL comparison CSV and plot")
print("\nDeep Learning training completed successfully!")
