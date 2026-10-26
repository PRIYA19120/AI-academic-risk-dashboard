import hashlib
import joblib
import pandas as pd

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "academic_data.csv"
MODEL_PATH = BASE_DIR / "model" / "academic_risk_model.pkl"

FEATURES = [
    "Attendance",
    "Assignment",
    "Quiz",
    "Internal",
    "Exam",
    "Final_Marks"
]

REQUIRED_COLUMNS = ["Student_ID"] + FEATURES


def get_dataset_signature():
    content = DATA_PATH.read_bytes()
    return hashlib.sha256(content).hexdigest()


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data["Student_ID"] = (
        data["Student_ID"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data = data[
        (data["Student_ID"] != "")
        & (data["Student_ID"].str.lower() != "nan")
    ].copy()

    for column in FEATURES:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data[FEATURES] = data[FEATURES].replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    data = data.dropna(subset=FEATURES, how="all")

    if len(data) < 2:
        raise ValueError(
            "At least two valid records are required."
        )

    if data[FEATURES].notna().sum().sum() == 0:
        raise ValueError(
            "No valid academic metric values found."
        )

    return data


def main():
    df = load_dataset()

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "detector",
            IsolationForest(
                n_estimators=200,
                contamination=0.10,
                random_state=42
            )
        )
    ])

    model.fit(df[FEATURES])

    predictions = model.predict(df[FEATURES])
    anomaly_scores = model.decision_function(df[FEATURES])

    df["Academic_Risk_Flag"] = [
        "Potential Anomaly" if prediction == -1
        else "No Anomaly Detected"
        for prediction in predictions
    ]

    print("\nModel training completed!")
    print("Training records:", len(df))
    print("Unique students:", df["Student_ID"].nunique())

    print("\nAnomaly detection results:")
    print(df["Academic_Risk_Flag"].value_counts())

    MODEL_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "model_type": "IsolationForest",
            "training_records": len(df),
            "dataset_signature": get_dataset_signature()
        },
        MODEL_PATH
    )

    print("\nModel saved successfully!")
    print("Saved model to:", MODEL_PATH)


if __name__ == "__main__":
    main()