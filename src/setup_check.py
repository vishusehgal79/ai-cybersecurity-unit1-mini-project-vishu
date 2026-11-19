import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn as sns

print("=" * 45)
print("AI CYBERSECURITY PROJECT - ENVIRONMENT CHECK")
print("=" * 45)

print("Python environment: Ready")
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Seaborn version:", sns.__version__)

sample = pd.DataFrame({"failed_attempts": [0, 1, 5, 8], "suspicious": [0, 0, 1, 1]})

print("\nSample security data:")
print(sample)

print("\nEnvironment validation successful!")
