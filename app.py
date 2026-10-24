
from flask import Flask, render_template, request
import pandas as pd
import joblib

from pathlib import Path

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
    course.lower(): course
    for course in course_order
}


def assign_support(row):
    score = 0
    valid_values = 0

    for column in FEATURES:
        value = pd.to_numeric(
            row[column], errors="coerce"
        )

        if pd.isna(value):
            continue

        valid_values += 1

        if value < 55:
            score += 2
        elif value < 70:
            score += 1

    if valid_values == 0:
        return "Moderate Support"

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

    required_columns = [
        "Student_ID",
        "Student_Name",
        "Course"
    ] + FEATURES

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV mein ye required columns missing hain: "
            + ", ".join(missing_columns)
        )

    for column in ["Student_ID", "Student_Name", "Course"]:
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
        [float("inf"), float("-inf")],
        float("nan")
    )

    data["Course"] = data["Course"].apply(
        lambda course: course_names.get(
            course.lower(), course
        )
    )

    if data.empty:
        raise ValueError(
            "CSV mein koi valid student record nahi mila."
        )

    return data


def load_ml_model():
    if not MODEL_PATH.exists():
        print(
            "WARNING: Trained model nahi mila. "
            "Abhi rule-based fallback use hoga."
        )
        return None

    try:
        saved_data = joblib.load(MODEL_PATH)

        model = saved_data["model"]
        features = saved_data.get("features", FEATURES)

        if not all(
            feature in FEATURES for feature in features
        ):
            print(
                "WARNING: Model features match nahi karte. "
                "Rule-based fallback use hoga."
            )
            return None

        return {
            "model": model,
            "features": features
        }

    except Exception as error:
        print("WARNING: Model load nahi hua:", error)
        print("Rule-based fallback use hoga.")
        return None


df = load_dataset()
ml_model = load_ml_model()


def update_support_levels():
    global df

    if ml_model is not None:
        try:
            features = ml_model["features"]
            predictions = ml_model["model"].predict(
                df[features]
            )

            df["Support_Level"] = predictions

            print("Support levels: ML model se predict hue.")

        except Exception as error:
            print("WARNING: ML prediction failed:", error)

            df["Support_Level"] = df.apply(
                assign_support, axis=1
            )

            print("Rule-based fallback use hua.")

    else:
        df["Support_Level"] = df.apply(
            assign_support, axis=1
        )

        print("Support levels: Rule-based fallback.")

    return df


update_support_levels()


def get_support_counts(support_series):
    return {
        "low": int(
            (support_series == "Low Support Need").sum()
        ),
        "moderate": int(
            (support_series == "Moderate Support").sum()
        ),
        "further": int(
            (support_series == "Further Review").sum()
        )
    }


def get_course_support_counts():
    return get_support_counts(df["Support_Level"])


def get_model_evaluation():
    model_accuracy = None
    evaluation_report = {}

    try:
        saved_model = joblib.load(MODEL_PATH)

        if "test_accuracy" in saved_model:
            model_accuracy = round(
                saved_model["test_accuracy"] * 100, 2
            )

        evaluation_report = saved_model.get(
            "classification_report", {}
        )

    except (FileNotFoundError, KeyError, ValueError, TypeError) as error:
        print("Model evaluation data unavailable:", error)

    return model_accuracy, evaluation_report


@app.route("/")
def dashboard():
    total_students = int(df["Student_ID"].nunique())
    courses_count = int(df["Course"].nunique())

    average_attendance = round(
        df["Attendance"].mean(), 1
    )
    average_marks = round(
        df["Final_Marks"].mean(), 1
    )
    average_assignment = round(
        df["Assignment"].mean(), 1
    )
    average_quiz = round(df["Quiz"].mean(), 1)
    average_internal = round(df["Internal"].mean(), 1)
    average_exam = round(df["Exam"].mean(), 1)

    counts = get_course_support_counts()

    review_df = df[
        df["Support_Level"] == "Further Review"
    ].copy()

    review_students = (
        review_df.sort_values(
            "Final_Marks", na_position="last"
        )
        .drop_duplicates(subset=["Student_ID"])
        .head(10)
        .to_dict(orient="records")
    )

    return render_template(
        "index.html",
        total_students=total_students,
        average_attendance=average_attendance,
        average_marks=average_marks,
        average_assignment=average_assignment,
        average_quiz=average_quiz,
        average_internal=average_internal,
        average_exam=average_exam,
        courses_count=courses_count,
        low_support=counts["low"],
        moderate_support=counts["moderate"],
        further_review=counts["further"],
        review_students=review_students
    )


@app.route("/courses")
def courses():
    course_data = {}

    for course in course_order:
        course_df = df[df["Course"] == course].copy()

        if course_df.empty:
            continue

        course_data[course] = {
            "students": int(
                course_df["Student_ID"].nunique()
            ),
            "avg_marks": round(
                course_df["Final_Marks"].mean(), 1
            ),
            "attendance": round(
                course_df["Attendance"].mean(), 1
            )
        }

    return render_template(
        "courses.html",
        course_data=course_data
    )


@app.route("/course/<course_slug>")
def course_details(course_slug):
    course_name = course_map.get(course_slug)

    if course_name is None:
        return (
            "Course not found. Please check the course link.",
            404
        )

    course_df = df[
        df["Course"] == course_name
    ].copy()

    if course_df.empty:
        return (
            f"No student records found for {course_name}.",
            404
        )

    total_students = int(
        course_df["Student_ID"].nunique()
    )
    average_marks = round(
        course_df["Final_Marks"].mean(), 1
    )
    average_attendance = round(
        course_df["Attendance"].mean(), 1
    )

    students = (
        course_df.sort_values("Student_Name")
        .drop_duplicates(subset=["Student_ID"])
        .to_dict(orient="records")
    )

    return render_template(
        "course_details.html",
        course_name=course_name,
        total_students=total_students,
        average_marks=average_marks,
        average_attendance=average_attendance,
        students=students
    )


@app.route("/student/<student_id>")
def student_details(student_id):
    student_id = str(student_id).strip()

    student_data = df[
        df["Student_ID"] == student_id
    ].copy()

    if student_data.empty:
        return (
            "Student not found. Please check the Student ID.",
            404
        )

    student_name = student_data.iloc[0]["Student_Name"]

    student_data["Course"] = pd.Categorical(
        student_data["Course"],
        categories=course_order,
        ordered=True
    )

    student_data = student_data.sort_values("Course")
    student_data = student_data.to_dict(
        orient="records"
    )

    return render_template(
        "student_details.html",
        student_id=student_id,
        student_name=student_name,
        student_data=student_data,
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
            "students": int(
                course_df["Student_ID"].nunique()
            ),
            "avg_marks": round(
                course_df["Final_Marks"].mean(), 1
            ),
            "attendance": round(
                course_df["Attendance"].mean(), 1
            ),
            "assignment": round(
                course_df["Assignment"].mean(), 1
            ),
            "quiz": round(
                course_df["Quiz"].mean(), 1
            )
        })

    counts = get_course_support_counts()

    student_report_data = df[
        [
            "Student_ID",
            "Student_Name",
            "Course",
            "Attendance",
            "Final_Marks",
            "Support_Level"
        ]
    ].copy()

    student_report_data = student_report_data.sort_values(
        ["Student_Name", "Course"]
    ).to_dict(orient="records")

    model_accuracy, evaluation_report = get_model_evaluation()

    return render_template(
        "reports.html",
        report_data=report_data,
        total_students=int(
            df["Student_ID"].nunique()
        ),
        total_courses=int(df["Course"].nunique()),
        overall_marks=round(
            df["Final_Marks"].mean(), 1
        ),
        overall_attendance=round(
            df["Attendance"].mean(), 1
        ),
        low_support_count=counts["low"],
        moderate_support_count=counts["moderate"],
        further_review_count=counts["further"],
        course_low_count=counts["low"],
        course_moderate_count=counts["moderate"],
        course_further_count=counts["further"],
        student_report_data=student_report_data,
        model_accuracy=model_accuracy,
        evaluation_report=evaluation_report
    )


@app.route("/students")
def all_students():
    search_query = request.args.get(
        "q", ""
    ).strip()

    selected_course = request.args.get(
        "course", ""
    ).strip()

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
                search_query,
                case=False,
                na=False,
                regex=False
            )
            |
            student_df["Student_Name"].str.contains(
                search_query,
                case=False,
                na=False,
                regex=False
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


if __name__ == "__main__":
    app.run(debug=True)
