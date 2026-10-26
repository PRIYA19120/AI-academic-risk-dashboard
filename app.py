from flask import Flask, render_template, request
from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
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

course_order = [
    "Machine Learning",
    "DBMS",
    "Cloud Computing",
    "Data Structures",
    "Python",
    "Data Science"
]

course_map = {
    "machine-learning": "Machine Learning",
    "dbms": "DBMS",
    "cloud-computing": "Cloud Computing",
    "data-structures": "Data Structures",
    "python": "Python",
    "data-science": "Data Science"
}

course_names = {
    course.lower(): course for course in course_order
}

df = pd.DataFrame()
ml_model = None
dataset_signature = None


def get_dataset_signature():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_PATH}"
        )

    content = DATA_PATH.read_bytes()
    return hashlib.sha256(content).hexdigest()


def load_dataset():
    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "Student_ID",
        "Student_Name",
        "Course"
    ] + FEATURES

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "CSV mein required columns missing hain: "
            + ", ".join(missing)
        )

    for column in [
        "Student_ID", "Student_Name", "Course"
    ]:
        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    data = data[
        (data["Student_ID"] != "")
        & (data["Student_Name"] != "")
        & (data["Course"] != "")
    ].copy()

    for column in FEATURES:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data[FEATURES] = data[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    data["Course"] = data["Course"].apply(
        lambda course: course_names.get(
            course.lower(), course
        )
    )

    if data.empty:
        raise ValueError("CSV mein valid records nahi hain.")

    if data[FEATURES].notna().sum().sum() == 0:
        raise ValueError(
            "CSV mein koi valid academic metric nahi hai."
        )

    return data


def train_anomaly_model(data):
    global ml_model

    if len(data) < 2:
        raise ValueError(
            "ML model train karne ke liye kam se kam "
            "2 valid records chahiye."
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=200,
            contamination=0.10,
            random_state=42
        ))
    ])

    model.fit(data[FEATURES])

    MODEL_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    ml_model = model

    joblib.dump({
        "model": model,
        "features": FEATURES,
        "model_type": "IsolationForest",
        "dataset_signature": get_dataset_signature()
    }, MODEL_PATH)

    print("Anomaly detection model trained and saved.")


def update_predictions():
    global df

    if ml_model is None:
        raise RuntimeError("ML model available nahi hai.")

    features = df[FEATURES]

    predictions = ml_model.predict(features)
    anomaly_scores = ml_model.decision_function(features)

    df["Anomaly_Score"] = anomaly_scores
    df["Academic_Risk_Flag"] = np.where(
        predictions == -1,
        "Potential Anomaly",
        "No Anomaly Detected"
    )

    # Lower anomaly score means a more unusual record.
    # These are relative categories, not verified support needs.
    low_threshold = np.quantile(anomaly_scores, 0.10)
    moderate_threshold = np.quantile(anomaly_scores, 0.30)

    df["Support_Level"] = np.select(
        [
            anomaly_scores <= low_threshold,
            anomaly_scores <= moderate_threshold
        ],
        [
            "Further Review",
            "Moderate Support"
        ],
        default="Low Support Need"
    )

    print("Predictions updated using Isolation Forest.")
    return df


def refresh_dataset():
    global df, dataset_signature

    current_signature = get_dataset_signature()

    if current_signature != dataset_signature:
        print("Dataset change detected. Refreshing model...")

        new_data = load_dataset()

        # Train on the latest CSV whenever its contents change.
        train_anomaly_model(new_data)

        df = new_data
        dataset_signature = current_signature

        update_predictions()


def initialize_app():
    global df, dataset_signature, ml_model

    df = load_dataset()
    signature = get_dataset_signature()

    # Train a fresh anomaly model on the current dataset.
    # This also replaces an older rule-label classifier.
    train_anomaly_model(df)

    dataset_signature = signature
    update_predictions()


@app.before_request
def refresh_before_request():
    refresh_dataset()


def get_support_counts(series):
    return {
        "low": int((series == "Low Support Need").sum()),
        "moderate": int((series == "Moderate Support").sum()),
        "further": int((series == "Further Review").sum())
    }


@app.route("/")
def dashboard():
    counts = get_support_counts(df["Support_Level"])

    review_students = (
        df[df["Support_Level"] == "Further Review"]
        .sort_values("Anomaly_Score")
        .drop_duplicates(subset=["Student_ID"])
        .head(10)
        .to_dict(orient="records")
    )

    return render_template(
        "index.html",
        total_students=int(df["Student_ID"].nunique()),
        courses_count=int(df["Course"].nunique()),
        average_attendance=round(df["Attendance"].mean(), 1),
        average_marks=round(df["Final_Marks"].mean(), 1),
        average_assignment=round(df["Assignment"].mean(), 1),
        average_quiz=round(df["Quiz"].mean(), 1),
        average_internal=round(df["Internal"].mean(), 1),
        average_exam=round(df["Exam"].mean(), 1),
        low_support=counts["low"],
        moderate_support=counts["moderate"],
        further_review=counts["further"],
        review_students=review_students
    )


@app.route("/courses")
def courses():
    course_data = {}

    for course in course_order:
        course_df = df[df["Course"] == course]

        if course_df.empty:
            continue

        course_data[course] = {
            "students": int(course_df["Student_ID"].nunique()),
            "avg_marks": round(course_df["Final_Marks"].mean(), 1),
            "attendance": round(course_df["Attendance"].mean(), 1)
        }

    return render_template(
        "courses.html",
        course_data=course_data
    )


@app.route("/course/<course_slug>")
def course_details(course_slug):
    course_name = course_map.get(course_slug)

    if course_name is None:
        return "Course not found.", 404

    course_df = df[df["Course"] == course_name].copy()

    if course_df.empty:
        return "No student records found for this course.", 404

    students = (
        course_df.sort_values("Student_Name")
        .drop_duplicates(subset=["Student_ID"])
        .to_dict(orient="records")
    )

    return render_template(
        "course_details.html",
        course_name=course_name,
        total_students=int(course_df["Student_ID"].nunique()),
        average_marks=round(course_df["Final_Marks"].mean(), 1),
        average_attendance=round(course_df["Attendance"].mean(), 1),
        students=students
    )


@app.route("/student/<student_id>")
def student_details(student_id):
    student_data = df[
        df["Student_ID"] == str(student_id).strip()
    ].copy()

    if student_data.empty:
        return "Student not found.", 404

    student_name = student_data.iloc[0]["Student_Name"]

    student_data["Course"] = pd.Categorical(
        student_data["Course"],
        categories=course_order,
        ordered=True
    )

    student_data = student_data.sort_values("Course")

    return render_template(
        "student_details.html",
        student_id=student_id,
        student_name=student_name,
        student_data=student_data.to_dict(orient="records"),
        course_order=course_order
    )


@app.route("/reports")
def reports():
    report_data = []

    for course in course_order:
        course_df = df[df["Course"] == course]

        if course_df.empty:
            continue

        report_data.append({
            "course": course,
            "students": int(course_df["Student_ID"].nunique()),
            "avg_marks": round(course_df["Final_Marks"].mean(), 1),
            "attendance": round(course_df["Attendance"].mean(), 1),
            "assignment": round(course_df["Assignment"].mean(), 1),
            "quiz": round(course_df["Quiz"].mean(), 1)
        })

    counts = get_support_counts(df["Support_Level"])

    student_report_data = df[
        [
            "Student_ID",
            "Student_Name",
            "Course",
            "Attendance",
            "Final_Marks",
            "Support_Level",
            "Academic_Risk_Flag"
        ]
    ].sort_values(
        ["Student_Name", "Course"]
    ).to_dict(orient="records")

    return render_template(
        "reports.html",
        report_data=report_data,
        total_students=int(df["Student_ID"].nunique()),
        total_courses=int(df["Course"].nunique()),
        overall_marks=round(df["Final_Marks"].mean(), 1),
        overall_attendance=round(df["Attendance"].mean(), 1),
        low_support_count=counts["low"],
        moderate_support_count=counts["moderate"],
        further_review_count=counts["further"],
        course_low_count=counts["low"],
        course_moderate_count=counts["moderate"],
        course_further_count=counts["further"],
        student_report_data=student_report_data,
        model_accuracy=None,
        evaluation_report={},
        anomaly_model=True
    )


@app.route("/students")
def all_students():
    search_query = request.args.get("q", "").strip()
    selected_course = request.args.get("course", "").strip()

    student_df = df[
        ["Student_ID", "Student_Name", "Course"]
    ].drop_duplicates(
        subset=["Student_ID", "Course"]
    )

    if selected_course in course_order:
        student_df = student_df[
            student_df["Course"] == selected_course
        ]

    if search_query:
        student_df = student_df[
            student_df["Student_ID"].str.contains(
                search_query, case=False, na=False, regex=False
            )
            |
            student_df["Student_Name"].str.contains(
                search_query, case=False, na=False, regex=False
            )
        ]

    students = (
        student_df.sort_values("Student_Name")
        .drop_duplicates(subset=["Student_ID"])
        .to_dict(orient="records")
    )

    return render_template(
        "students.html",
        students=students,
        course_order=course_order,
        search_query=search_query,
        selected_course=selected_course
    )


initialize_app()

if __name__ == "__main__":
    app.run(debug=True)