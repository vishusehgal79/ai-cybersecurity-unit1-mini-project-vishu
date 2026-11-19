import json
import pandas as pd

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results"
PLOTS = ROOT / "outputs" / "plots"
REPORT_DIR = ROOT / "report"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = REPORT_DIR / "AI_Cybersecurity_Threat_Analysis_Report.pdf"

# Load model metrics
with open(RESULTS / "logistic_regression_metrics.json") as f:
    lr = json.load(f)

with open(RESULTS / "mlp_metrics.json") as f:
    mlp = json.load(f)

threshold_df = pd.read_csv(RESULTS / "threshold_analysis.csv")
group_df = pd.read_csv(RESULTS / "group_fairness_analysis.csv")

# Styles
styles = getSampleStyleSheet()

styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=21,
        leading=27,
        alignment=TA_CENTER,
        spaceAfter=14,
    )
)

styles.add(
    ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading1"],
        fontSize=15,
        leading=19,
        spaceBefore=10,
        spaceAfter=7,
        textColor=colors.HexColor("#163A5F"),
    )
)

styles.add(
    ParagraphStyle(
        name="SmallText", parent=styles["BodyText"], fontSize=8.5, leading=12
    )
)

doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=A4,
    rightMargin=40,
    leftMargin=40,
    topMargin=40,
    bottomMargin=40,
)

story = []


def heading(text):
    story.append(Paragraph(text, styles["SectionHeading"]))


def paragraph(text):
    story.append(Paragraph(text, styles["BodyText"]))
    story.append(Spacer(1, 6))


def make_table(data, widths=None):
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163A5F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F0F4F8")],
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 8))


def add_plot(filename, width=245):
    path = PLOTS / filename
    if path.exists():
        img = Image(str(path))
        img.drawWidth = width
        img.drawHeight = width * 0.66
        return img
    return Paragraph(f"Plot unavailable: {filename}", styles["SmallText"])


def add_plot_pair(file1, file2):
    images = Table([[add_plot(file1), add_plot(file2)]], colWidths=[250, 250])
    images.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(images)
    story.append(Spacer(1, 10))


# --------------------------------------------------
# PAGE 1 - OVERVIEW
# --------------------------------------------------

story.append(
    Paragraph("AI Cybersecurity Threat Landscape Analysis", styles["ReportTitle"])
)

story.append(
    Paragraph("Unit 1 Mini Project | Login Activity Classification", styles["Normal"])
)
story.append(Spacer(1, 14))

heading("1. Project Overview")

paragraph(
    "This project explores the use of machine learning and a neural network "
    "prototype to classify login activity as normal or suspicious. It includes "
    "dataset generation, exploratory data analysis, data preprocessing, "
    "classification, performance evaluation and ethical analysis."
)

heading("2. Objectives")

paragraph(
    "The objectives are to analyze login activity patterns, identify "
    "potentially suspicious behavior, train classification models, compare "
    "their performance, and investigate the impact of classification "
    "thresholds and group-level flag rates."
)

heading("3. Dataset")

paragraph(
    "A reproducible synthetic dataset of 5,000 login records was created. "
    "The dataset contains login hour, failed attempts, IP risk score, "
    "unusual location indicator, device trust status, account age, "
    "login success, region group and the suspicious activity target label."
)

make_table(
    [
        ["Dataset property", "Value"],
        ["Total records", "5,000"],
        ["Total features", "8 input features"],
        ["Target", "is_suspicious"],
        ["Normal records", "4,069 (81.38%)"],
        ["Suspicious records", "931 (18.62%)"],
        ["Missing values", "None detected"],
        ["Data source", "Synthetic / generated for this project"],
    ],
    [190, 310],
)

heading("4. Methodology")

paragraph(
    "The dataset was inspected and visualized using Pandas, Matplotlib "
    "and Seaborn. Numeric features were imputed using the median and "
    "standardized using StandardScaler. The region group feature was "
    "imputed using the most frequent category and encoded using one-hot "
    "encoding. Both models used an 80/20 stratified train-test split "
    "with random_state=42."
)

story.append(PageBreak())

# --------------------------------------------------
# PAGE 2 - EDA AND MODELS
# --------------------------------------------------

heading("5. Exploratory Data Analysis")

paragraph(
    "Exploratory analysis examined the target distribution, failed login "
    "attempts, IP risk scores, login hour patterns, unusual location "
    "activity and feature correlations. The synthetic data showed a "
    "suspicious activity rate of 18.62%. Suspicious rates were higher "
    "among records with unusual locations, untrusted devices and failed "
    "login outcomes. These relationships are part of the generated data "
    "and should not be interpreted as real-world findings."
)

add_plot_pair("01_threat_distribution.png", "02_failed_attempts.png")
add_plot_pair("03_ip_risk_distribution.png", "04_suspicious_by_hour.png")
add_plot_pair("05_location_analysis.png", "06_correlation_heatmap.png")

story.append(PageBreak())

heading("6. Machine Learning Model")

paragraph(
    "Logistic Regression was used as the baseline classifier. It provides "
    "a relatively simple and fast approach to binary classification and "
    "serves as a baseline for evaluating the neural network model."
)

heading("7. Deep Learning Prototype")

paragraph(
    "An MLP Neural Network was trained using hidden layer sizes of "
    "(32, 16), ReLU activation, the Adam optimizer and early stopping. "
    "The model was implemented using Scikit-learn's MLPClassifier."
)

heading("8. Model Performance")

model_table = [
    ["Metric", "Logistic Regression", "MLP"],
    ["Accuracy", f"{lr['accuracy']:.4f}", f"{mlp['accuracy']:.4f}"],
    ["Precision", f"{lr['precision']:.4f}", f"{mlp['precision']:.4f}"],
    ["Recall", f"{lr['recall']:.4f}", f"{mlp['recall']:.4f}"],
    ["F1-score", f"{lr['f1_score']:.4f}", f"{mlp['f1_score']:.4f}"],
    [
        "Training time (seconds)",
        f"{lr['training_time_seconds']:.4f}",
        f"{mlp['training_time_seconds']:.4f}",
    ],
]

make_table(model_table, [145, 175, 175])

paragraph(
    "The MLP achieved higher accuracy and precision, and a slightly "
    "higher F1-score. Logistic Regression achieved slightly higher recall "
    "and required less training time. Both models showed low recall, "
    "indicating that many suspicious records were missed."
)

add_plot_pair("07_logistic_confusion_matrix.png", "08_mlp_confusion_matrix.png")
add_plot_pair("09_model_comparison.png", "10_threshold_tradeoff.png")

story.append(PageBreak())

# --------------------------------------------------
# PAGE 3 - ETHICS
# --------------------------------------------------

heading("9. Decision Threshold Experiment")

paragraph(
    "The Logistic Regression probability threshold was adjusted to 0.25, "
    "0.50 and 0.75. Lowering the threshold increased detection recall "
    "but also increased false positives. Raising the threshold reduced "
    "false positives while increasing missed suspicious activity."
)

threshold_table = [["Threshold", "Accuracy", "Precision", "Recall", "FP", "FN"]]

for _, row in threshold_df.iterrows():
    threshold_table.append(
        [
            str(row["Threshold"]),
            f'{row["Accuracy"]:.4f}',
            f'{row["Precision"]:.4f}',
            f'{row["Recall"]:.4f}',
            str(int(row["False Positives"])),
            str(int(row["False Negatives"])),
        ]
    )

make_table(threshold_table, [65, 70, 70, 70, 55, 55])

paragraph(
    "At threshold 0.25, recall was 53.23% and there were 144 false "
    "positives. At threshold 0.75, recall dropped to 2.69%, with only "
    "3 false positives. The results illustrate the trade-off between "
    "detecting threats and generating false alarms."
)

heading("10. Group-Level Analysis")

group_table = [["Group", "Records", "Actual suspicious rate", "Model flag rate"]]

for _, row in group_df.iterrows():
    group_table.append(
        [
            str(row["region_group"]),
            str(int(row["Total_Records"])),
            f'{row["Actual_Suspicious_Rate"]:.2%}',
            f'{row["Predicted_Flag_Rate"]:.2%}',
        ]
    )

make_table(group_table, [90, 70, 150, 140])

paragraph(
    "The model flag rates differed between Region_A and Region_B. "
    "This difference is not sufficient to establish bias because the "
    "dataset is synthetic and group characteristics and sample sizes "
    "may differ. Group-level testing is an important part of responsible "
    "model evaluation."
)

add_plot("11_group_flag_rates.png", width=390)

heading("11. Ethical Considerations")

paragraph(
    "<b>False positives:</b> Legitimate users may be incorrectly "
    "flagged, potentially resulting in inconvenience or unnecessary "
    "account restrictions."
)

paragraph(
    "<b>False negatives:</b> Suspicious activity may remain undetected, "
    "creating security risks."
)

paragraph(
    "<b>Bias and fairness:</b> Models should be evaluated across relevant "
    "groups. Differences in flag rates should be investigated and should "
    "not be interpreted without considering the underlying data."
)

paragraph(
    "<b>Privacy:</b> Login data should be minimized, protected and "
    "processed only for legitimate, authorized security purposes."
)

paragraph(
    "<b>Human oversight:</b> High-impact actions such as account "
    "blocking should include additional verification or human review "
    "rather than relying solely on an automated prediction."
)

story.append(PageBreak())

# --------------------------------------------------
# PAGE 4 - CONCLUSION
# --------------------------------------------------

heading("12. Limitations")

paragraph(
    "This project uses synthetic data rather than real network or login "
    "records. Results therefore demonstrate an experimental workflow "
    "and cannot establish real-world cybersecurity effectiveness. "
    "The dataset is relatively small and generated with predefined "
    "patterns. Both models showed limited suspicious-class recall. "
    "Further validation with representative real-world data, careful "
    "feature engineering, threshold selection and testing would be "
    "required before deployment."
)

heading("13. Conclusion")

paragraph(
    "The mini-project implemented an end-to-end cybersecurity "
    "classification workflow, including synthetic data creation, "
    "exploratory analysis, preprocessing, Logistic Regression and "
    "MLP-based classification, model comparison and ethical evaluation. "
    "The MLP produced a slightly higher accuracy and F1-score, while "
    "Logistic Regression trained faster and had slightly higher recall. "
    "The threshold experiment demonstrated that the detection rate and "
    "false alarms can change substantially depending on the decision "
    "threshold. The results emphasize that cybersecurity systems should "
    "consider recall, false positives, fairness, privacy and human "
    "oversight in addition to accuracy."
)

heading("14. Tools & Technologies")

paragraph(
    "Python 3.11, Pandas, NumPy, Scikit-learn, Matplotlib, "
    "Seaborn, Joblib, ReportLab and Git."
)

heading("Project Disclaimer")

paragraph(
    "This is an educational project using synthetic login activity "
    "data. It is not a production intrusion detection or authentication "
    "system and should not be used to make real account security decisions."
)

# Build the PDF
doc.build(story)

print("=" * 55)
print("REPORT GENERATED SUCCESSFULLY")
print("=" * 55)
print(f"Report: {PDF_PATH}")
print(f"File exists: {PDF_PATH.exists()}")
print(f"File size: {PDF_PATH.stat().st_size:,} bytes")
