import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "login_activity.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
PLOTS_DIR = ROOT / "outputs" / "plots"

model_path = RESULTS_DIR / "logistic_regression_model.joblib"

# Load dataset and trained model
df = pd.read_csv(DATA_PATH)
model = joblib.load(model_path)

X = df.drop(columns=["is_suspicious"])
y = df["is_suspicious"]

# Same split used in previous experiments
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Probability of being suspicious
probabilities = model.predict_proba(X_test)[:, 1]

# Experiment A: Threshold analysis
thresholds = [0.25, 0.50, 0.75]
threshold_results = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    threshold_results.append(
        {
            "Threshold": threshold,
            "Accuracy": round(accuracy_score(y_test, predictions), 4),
            "Precision": round(
                precision_score(y_test, predictions, zero_division=0), 4
            ),
            "Recall": round(recall_score(y_test, predictions, zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, predictions, zero_division=0), 4),
            "False Positives": int(fp),
            "False Negatives": int(fn),
            "True Positives": int(tp),
            "True Negatives": int(tn),
        }
    )

threshold_df = pd.DataFrame(threshold_results)

print("=" * 65)
print("ETHICAL ANALYSIS - DECISION THRESHOLD EXPERIMENT")
print("=" * 65)
print(threshold_df.to_string(index=False))

threshold_df.to_csv(RESULTS_DIR / "threshold_analysis.csv", index=False)

# Plot precision and recall across thresholds
plt.figure(figsize=(8, 5))
plt.plot(
    threshold_df["Threshold"], threshold_df["Precision"], marker="o", label="Precision"
)
plt.plot(threshold_df["Threshold"], threshold_df["Recall"], marker="o", label="Recall")
plt.plot(
    threshold_df["Threshold"], threshold_df["F1 Score"], marker="o", label="F1 Score"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Metric Score")
plt.title("Effect of Decision Threshold on Model Performance")
plt.xticks(thresholds)
plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(PLOTS_DIR / "10_threshold_tradeoff.png", dpi=300)
plt.close()

# Experiment B: Group analysis
# region_group is a synthetic grouping variable, not a sensitive attribute.
group_data = X_test.copy()
group_data["Actual"] = y_test.values
group_data["Predicted"] = (probabilities >= 0.50).astype(int)

group_results = (
    group_data.groupby("region_group")
    .agg(
        Total_Records=("Actual", "count"),
        Actual_Suspicious_Rate=("Actual", "mean"),
        Predicted_Flag_Rate=("Predicted", "mean"),
    )
    .reset_index()
)

group_results["Actual_Suspicious_Rate"] = group_results["Actual_Suspicious_Rate"].round(
    4
)
group_results["Predicted_Flag_Rate"] = group_results["Predicted_Flag_Rate"].round(4)

print("\n" + "=" * 65)
print("GROUP ANALYSIS - REGION GROUP")
print("=" * 65)
print(group_results.to_string(index=False))

group_results.to_csv(RESULTS_DIR / "group_fairness_analysis.csv", index=False)

# Plot group flag rates
plt.figure(figsize=(8, 5))

group_results.set_index("region_group")[
    ["Actual_Suspicious_Rate", "Predicted_Flag_Rate"]
].plot(kind="bar", figsize=(8, 5))

plt.title("Actual Suspicious Rate vs Model Flag Rate by Group")
plt.ylabel("Rate")
plt.xlabel("Region Group")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(PLOTS_DIR / "11_group_flag_rates.png", dpi=300)
plt.close()

print("\nSaved outputs:")
print("- threshold_analysis.csv")
print("- group_fairness_analysis.csv")
print("- 10_threshold_tradeoff.png")
print("- 11_group_flag_rates.png")
print("\nEthical analysis completed successfully!")
