import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "login_activity.csv"
PLOT_DIR = ROOT / "outputs" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")
print("=" * 60)
print("CYBERSECURITY THREAT LANDSCAPE ANALYSIS")
print("=" * 60)

# Basic information
print("\nDataset shape:", df.shape)
print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Class distribution
counts = df["is_suspicious"].value_counts().sort_index()
labels = ["Normal", "Suspicious"]

print("\nThreat distribution:")
print(f"Normal activities: {counts.get(0, 0)}")
print(f"Suspicious activities: {counts.get(1, 0)}")

# 1. Normal vs suspicious activity
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="is_suspicious")
plt.xticks([0, 1], labels)
plt.title("Normal vs Suspicious Login Activities")
plt.xlabel("Activity Type")
plt.ylabel("Number of Login Events")
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_threat_distribution.png", dpi=300)
plt.close()

# 2. Failed attempts comparison
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="is_suspicious", y="failed_attempts")
plt.xticks([0, 1], labels)
plt.title("Failed Login Attempts by Activity Type")
plt.xlabel("Activity Type")
plt.ylabel("Failed Attempts")
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_failed_attempts.png", dpi=300)
plt.close()

# 3. IP risk score distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="ip_risk_score", hue="is_suspicious", bins=20, kde=True)
plt.title("IP Risk Score Distribution")
plt.xlabel("IP Risk Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_ip_risk_distribution.png", dpi=300)
plt.close()

# 4. Suspicious activity by login hour
hourly = df.groupby("login_hour")["is_suspicious"].sum()

plt.figure(figsize=(10, 5))
sns.barplot(x=hourly.index, y=hourly.values)
plt.title("Suspicious Login Activity by Hour")
plt.xlabel("Login Hour")
plt.ylabel("Suspicious Events")
plt.xticks(range(24))
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_suspicious_by_hour.png", dpi=300)
plt.close()

# 5. Unusual location comparison
location_analysis = df.groupby("unusual_location")["is_suspicious"].mean().reset_index()

plt.figure(figsize=(7, 5))
sns.barplot(data=location_analysis, x="unusual_location", y="is_suspicious")
plt.xticks([0, 1], ["Usual Location", "Unusual Location"])
plt.title("Suspicious Activity Rate by Location")
plt.xlabel("Location Type")
plt.ylabel("Suspicious Activity Rate")
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_location_analysis.png", dpi=300)
plt.close()

# 6. Feature correlation heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Cybersecurity Login Feature Correlation")
plt.tight_layout()
plt.savefig(PLOT_DIR / "06_correlation_heatmap.png", dpi=300)
plt.close()

# Suspicious activity rates for key indicators
print("\nSuspicious activity rates by key indicators:")

for feature in ["unusual_location", "device_trusted", "login_success"]:
    print(f"\n{feature}:")
    print(df.groupby(feature)["is_suspicious"].mean().round(3))

print("\nAll EDA charts saved to:", PLOT_DIR)
print("\nGenerated plots:")
for file in sorted(PLOT_DIR.glob("*.png")):
    print("-", file.name)

print("\nEDA completed successfully!")
