
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "academic_data.csv"
MODEL_PATH = BASE_DIR / "model" / "academic_risk_model.pkl"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Attendance",
    "Assignment",
    "Quiz",
    "Internal",
    "Exam",
    "Final_Marks"
]

REQUIRED_COLUMNS = ["Student_ID"] + FEATURES


def assign_support(row):
    score = 0

    for column in FEATURES:
        value = row[column]

        if pd.isna(value):
            continue

        if value < 55:
            score += 2
        elif value < 70:
            score += 1

    if score >= 6:
        return "Further Review"
    elif score >= 3:
        return "Moderate Support"
    else:
        return "Low Support Need"


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

    data["Student_ID"] = data["Student_ID"].astype(str).str.strip()

    data = data[
        data["Student_ID"].notna()
        & (data["Student_ID"] != "")
        & (data["Student_ID"].str.lower() != "nan")
    ].copy()

    for column in FEATURES:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data[FEATURES] = data[FEATURES].replace(
        [float("inf"), float("-inf")], float("nan")
    )

    data = data.dropna(subset=FEATURES, how="all")

    if len(data) < 2:
        raise ValueError(
            "Not enough valid records to train the model."
        )

    if data["Student_ID"].nunique() < 2:
        raise ValueError(
            "At least two different students are required "
            "for student-wise training and testing."
        )

    return data


def main():
    df = load_dataset()

    df["Support_Level"] = df.apply(
        assign_support, axis=1
    )

    print("\nSupport level counts:")
    print(df["Support_Level"].value_counts())

    X = df[FEATURES]
    y = df["Support_Level"]
    groups = df["Student_ID"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains fewer than two support "
            "classes. Add more varied student records."
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test, predictions
    )

    evaluation_report = classification_report(
        y_test,
        predictions,
        labels=[
            "Low Support Need",
            "Moderate Support",
            "Further Review"
        ],
        output_dict=True,
        zero_division=0
    )

    print("\nModel training completed!")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print(
        "Training students:",
        groups.iloc[train_indices].nunique()
    )

    print(
        "Testing students:",
        groups.iloc[test_indices].nunique()
    )

    print(
        "Test accuracy:",
        round(test_accuracy * 100, 2),
        "%"
    )

    print("\nClassification report:")

    print(classification_report(
        y_test,
        predictions,
        labels=[
            "Low Support Need",
            "Moderate Support",
            "Further Review"
        ],
        zero_division=0
    ))

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target_classes": model.classes_.tolist(),
            "training_records": len(X_train),
            "test_records": len(X_test),
            "test_accuracy": float(test_accuracy),
            "classification_report": evaluation_report,
            "label_method": "rule_based_six_metric_score"
        },
        MODEL_PATH
    )

    print("\nModel saved successfully!")
    print("Saved model to:", MODEL_PATH)


if __name__ == "__main__":
    main()
