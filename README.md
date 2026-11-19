# AI Cybersecurity Threat Landscape Analysis

## Overview
This project analyzes login activity to identify potentially suspicious behavior using Machine Learning and a Deep Learning prototype. It includes synthetic dataset generation, exploratory data analysis, classification, model comparison, and ethical evaluation.

## Objectives
- Explore login activity and identify suspicious patterns.
- Perform data preprocessing and feature analysis.
- Train Logistic Regression and an MLP Neural Network.
- Compare model performance using accuracy, precision, recall, F1-score, and training time.
- Examine false positives, false negatives, threshold effects, and group-level differences.

## Dataset
A synthetic dataset containing 5,000 login records with features such as login hour, failed attempts, IP risk score, device trust, location indicator, account age, login success, and region group.

**Target:** `is_suspicious` — Normal (0) or Suspicious (1).

## Technologies Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, ReportLab, Git.

## Models
- Logistic Regression
- MLP Neural Network with hidden layers (32, 16), ReLU activation, and Adam optimizer.

## Results
| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.80% | 45.45% | 16.13% | 23.81% |
| MLP Neural Network | 82.30% | 59.57% | 15.05% | 24.03% |

The MLP achieved slightly higher accuracy and F1-score, while Logistic Regression had slightly higher recall and lower training time.

## Ethical Considerations
- False positives can affect legitimate users.
- False negatives can allow suspicious activity to go undetected.
- Classification thresholds influence the balance between security alerts and false alarms.
- Group-level differences require investigation and should not automatically be interpreted as bias.
- User login data must be handled responsibly, with privacy protection and human oversight.

## Limitations
The project uses synthetic data and is intended for educational experimentation. Results do not represent validated real-world threat detection performance.

## Project Structure
- `data/` — dataset
- `src/` — scripts for generation, EDA, model training and report generation
- `outputs/plots/` — generated visualizations
- `outputs/results/` — model metrics, reports and trained models
- `report/` — final PDF report

## How to Run
```bash
pip install -r requirements.txt
python src/generate_dataset.py
python src/eda.py
python src/ml_model.py
python src/dl_model.py
python src/ethical_analysis.py
python src/create_report.py
```

## Conclusion
This project demonstrates an end-to-end AI-based cybersecurity analysis pipeline and highlights the importance of evaluating security models beyond accuracy, including recall, error trade-offs, ethical considerations, and responsible deployment.
