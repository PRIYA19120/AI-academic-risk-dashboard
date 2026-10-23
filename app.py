
from flask import Flask, render_template
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "academic_data.csv"
MODEL_PATH = BASE_DIR / "model" / "academic_risk_model.pkl"

df = pd.read_csv(DATA_PATH)

for column in ["Student_ID", "Student_Name", "Course"]:
    df[column] = df[column].astype(str).str.strip()

course_order = [
    "Machine Learning",
    "DBMS",
    "Cloud Computing",
    "Data Structures",
    "Python",
    "Data Science"
]

df["Course"] = df["Course"].replace({
    "machine learning": "Machine Learning",
    "dbms": "DBMS",
    "cloud computing": "Cloud Computing",
    "data structures": "Data Structures",
    "python": "Python",
    "data science": "Data Science"
})

course_map = {
    "machine-learning": "Machine Learning",
    "dbms": "DBMS",
    "cloud-computing": "Cloud Computing",
    "data-structures": "Data Structures",
    "python": "Python",
    "data-science": "Data Science"
}

model_data = joblib.load(MODEL_PATH)
ml_model = model_data["model"]
features = model_data["features"]

df["Support_Level"] = ml_model.predict(df[features])


@app.route("/")
def dashboard():
    total_students = int(df["Student_ID"].nunique())

    average_attendance = round(df["Attendance"].mean(), 1)
    average_marks = round(df["Final_Marks"].mean(), 1)
    average_assignment = round(df["Assignment"].mean(), 1)
    average_quiz = round(df["Quiz"].mean(), 1)
    average_internal = round(df["Internal"].mean(), 1)
    average_exam = round(df["Exam"].mean(), 1)
    courses_count = int(df["Course"].nunique())

    student_support = (
        df.groupby("Student_ID")["Support_Level"]
        .apply(
            lambda values:
                "Further Review"
                if "Further Review" in values.values
                else (
                    "Moderate Support"
                    if "Moderate Support" in values.values
                    else "Low Support Need"
                )
        )
    )

    low_support = int(
        (student_support == "Low Support Need").sum()
    )
    moderate_support = int(
        (student_support == "Moderate Support").sum()
    )
    further_review = int(
        (student_support == "Further Review").sum()
    )

    review_df = df[
        df["Support_Level"] == "Further Review"
    ].copy()

    review_students = (
        review_df.sort_values("Final_Marks")
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
        low_support=low_support,
        moderate_support=moderate_support,
        further_review=further_review,
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
            "students": int(course_df["Student_ID"].nunique()),
            "avg_marks": round(course_df["Final_Marks"].mean(), 1),
            "attendance": round(course_df["Attendance"].mean(), 1)
        }

    return render_template("courses.html", course_data=course_data)


@app.route("/course/<course_slug>")
def course_details(course_slug):
    course_name = course_map.get(course_slug)

    if course_name is None:
        return "Course not found. Please check the course link.", 404

    course_df = df[df["Course"] == course_name].copy()

    if course_df.empty:
        return f"No student records found for {course_name}.", 404

    total_students = int(course_df["Student_ID"].nunique())
    average_marks = round(course_df["Final_Marks"].mean(), 1)
    average_attendance = round(course_df["Attendance"].mean(), 1)

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
        return "Student not found. Please check the Student ID.", 404

    student_name = student_data.iloc[0]["Student_Name"]

    student_data["Course"] = pd.Categorical(
        student_data["Course"],
        categories=course_order,
        ordered=True
    )

    student_data = student_data.sort_values("Course")

    student_data = student_data.to_dict(orient="records")

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
            "students": int(course_df["Student_ID"].nunique()),
            "avg_marks": round(course_df["Final_Marks"].mean(), 1),
            "attendance": round(course_df["Attendance"].mean(), 1),
            "assignment": round(course_df["Assignment"].mean(), 1),
            "quiz": round(course_df["Quiz"].mean(), 1)
        })

    return render_template(
        "reports.html",
        report_data=report_data,
        total_students=int(df["Student_ID"].nunique()),
        total_courses=int(df["Course"].nunique()),
        overall_marks=round(df["Final_Marks"].mean(), 1),
        overall_attendance=round(df["Attendance"].mean(), 1)
    )


@app.route("/students")
def all_students():
    students = (
        df[["Student_ID", "Student_Name"]]
        .drop_duplicates(subset=["Student_ID"])
        .sort_values("Student_Name")
        .to_dict(orient="records")
    )

    return render_template("students.html", students=students)


if __name__ == "__main__":
    app.run(debug=True)
