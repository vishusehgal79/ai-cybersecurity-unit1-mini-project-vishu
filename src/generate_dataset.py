import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible dataset
np.random.seed(42)

# Number of login records
N = 5000

# Generate login features
login_hour = np.random.randint(0, 24, N)
failed_attempts = np.random.poisson(1.2, N)
ip_risk_score = np.random.randint(0, 101, N)
unusual_location = np.random.binomial(1, 0.12, N)
device_trusted = np.random.binomial(1, 0.82, N)
account_age_days = np.random.randint(1, 2000, N)
login_success = np.random.binomial(1, 0.92, N)

# Generate suspicious activity probability
risk = (
    (failed_attempts >= 4) * 0.35
    + (ip_risk_score >= 70) * 0.30
    + unusual_location * 0.20
    + (device_trusted == 0) * 0.15
    + ((login_hour <= 4) | (login_hour >= 23)) * 0.10
    + (login_success == 0) * 0.10
)

# Convert risk into suspicious/normal labels
probability = np.clip(risk, 0, 0.95)
is_suspicious = np.random.binomial(1, probability)

# Add a simulated group field for later ethical/fairness analysis.
# This is synthetic and does not represent real people.
region_group = np.random.choice(["Region_A", "Region_B"], size=N, p=[0.7, 0.3])

# Create dataset
df = pd.DataFrame(
    {
        "login_hour": login_hour,
        "failed_attempts": failed_attempts,
        "ip_risk_score": ip_risk_score,
        "unusual_location": unusual_location,
        "device_trusted": device_trusted,
        "account_age_days": account_age_days,
        "login_success": login_success,
        "region_group": region_group,
        "is_suspicious": is_suspicious,
    }
)

# Save dataset in project data directory
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
data_dir.mkdir(exist_ok=True)

output_path = data_dir / "login_activity.csv"
df.to_csv(output_path, index=False)

print("=" * 50)
print("CYBERSECURITY DATASET GENERATED")
print("=" * 50)
print(f"Total records: {len(df)}")
print(f"Dataset saved to: {output_path}")
print("\nDataset preview:")
print(df.head(10))
print("\nClass distribution:")
print(df["is_suspicious"].value_counts())
print("\nSuspicious activity percentage:")
print(round(df["is_suspicious"].mean() * 100, 2), "%")
print("\nMissing values:")
print(df.isnull().sum())
