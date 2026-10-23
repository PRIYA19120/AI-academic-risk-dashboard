
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "academic_data.csv"
MODEL_PATH = BASE_DIR / "model" / "academic_risk_model.pkl"

df = pd.read_csv(DATA_PATH)

# Generate training labels using the project's existing rules
def assign_support(row):
    score = 0

    if row["Attendance"] < 60:
        score += 2
    elif row["Attendance"] < 75:
        score += 1

    if row["Assignment"] < 50:
        score += 2
    elif row["Assignment"] < 70:
        score += 1

    if row["Final_Marks"] < 50:
        score += 2
    elif row["Final_Marks"] < 65:
        score += 1

    if score >= 4:
        return "Further Review"
    elif score >= 2:
        return "Moderate Support"
    return "Low Support Need"


df["Support_Level"] = df.apply(assign_support, axis=1)

features = [
    "Attendance",
    "Assignment",
    "Quiz",
    "Internal",
    "Exam",
    "Final_Marks"
]

X = df[features]
y = df["Support_Level"]

# Split records for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model training completed!")
print("Test accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(
    {
        "model": model,
        "features": features
    },
    MODEL_PATH
)

print("\nSaved model to:", MODEL_PATH)
