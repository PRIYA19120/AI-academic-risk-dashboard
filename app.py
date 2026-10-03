from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("dataset/academic_data.csv")


# Clean important text columns
df["Student_ID"] = df["Student_ID"].astype(str).str.strip()
df["Student_Name"] = df["Student_Name"].astype(str).str.strip()
df["Course"] = df["Course"].astype(str).str.strip()


# =========================================================
# COURSE ORDER
# =========================================================

course_order = [
    "Machine Learning",
    "DBMS",
    "Cloud Computing",
    "Data Structures",
    "Python",
    "Data Science"
]


# =========================================================
# ACADEMIC SUPPORT CALCULATION
# =========================================================

def calculate_support(row):

    score = 0

    # Attendance
    if row["Attendance"] < 60:
        score += 2
    elif row["Attendance"] < 75:
        score += 1

    # Assignment
    if row["Assignment"] < 50:
        score += 2
    elif row["Assignment"] < 70:
        score += 1

    # Final Marks
    if row["Final_Marks"] < 50:
        score += 2
    elif row["Final_Marks"] < 65:
        score += 1

    # Support category
    if score >= 4:
        return "Further Review"
    elif score >= 2:
        return "Moderate Support"
    else:
        return "Low Support Need"


# Apply support calculation
df["Support_Level"] = df.apply(
    calculate_support,
    axis=1
)


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/")
def dashboard():

    # Unique students
    total_students = df["Student_ID"].nunique()

    # Overall averages
    average_attendance = round(
        df["Attendance"].mean(), 1
    )

    average_marks = round(
        df["Final_Marks"].mean(), 1
    )

    average_assignment = round(
        df["Assignment"].mean(), 1
    )

    average_quiz = round(
        df["Quiz"].mean(), 1
    )

    average_internal = round(
        df["Internal"].mean(), 1
    )

    average_exam = round(
        df["Exam"].mean(), 1
    )

    courses_count = df["Course"].nunique()


    # =====================================================
    # STUDENT-LEVEL SUPPORT ANALYSIS
    # =====================================================

    student_support = (
        df.groupby("Student_ID")["Support_Level"]
        .apply(
            lambda x:
                "Further Review"
                if "Further Review" in x.values
                else (
                    "Moderate Support"
                    if "Moderate Support" in x.values
                    else "Low Support Need"
                )
        )
    )


    # Count unique students
    low_support = int(
        (
            student_support
            == "Low Support Need"
        ).sum()
    )

    moderate_support = int(
        (
            student_support
            == "Moderate Support"
        ).sum()
    )

    further_review = int(
        (
            student_support
            == "Further Review"
        ).sum()
    )


    # =====================================================
    # STUDENTS FOR FURTHER REVIEW
    # =====================================================

    review_df = df[
        df["Support_Level"]
        == "Further Review"
    ].copy()


    # One student should appear only once
    review_students = (
        review_df
        .sort_values("Final_Marks")
        .drop_duplicates(
            subset=["Student_ID"]
        )
        .head(10)
        .to_dict(
            orient="records"
        )
    )


    # =====================================================
    # SEND DATA TO DASHBOARD
    # =====================================================

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


# =========================================================
# COURSES PAGE
# =========================================================

@app.route("/courses")
def courses():

    course_data = {}


    for course in course_order:

        course_df = df[
            df["Course"] == course
        ]

        if course_df.empty:
            continue


        course_data[course] = {

            "students":
                course_df[
                    "Student_ID"
                ].nunique(),

            "avg_marks":
                round(
                    course_df[
                        "Final_Marks"
                    ].mean(),
                    1
                ),

            "attendance":
                round(
                    course_df[
                        "Attendance"
                    ].mean(),
                    1
                )
        }


    return render_template(
        "courses.html",
        course_data=course_data
    )


# =========================================================
# COURSE MAP
# =========================================================

course_map = {

    "machine-learning":
        "Machine Learning",

    "dbms":
        "DBMS",

    "cloud-computing":
        "Cloud Computing",

    "data-structures":
        "Data Structures",

    "python":
        "Python",

    "data-science":
        "Data Science"
}


# =========================================================
# COURSE DETAILS
# =========================================================

@app.route("/course/<course_slug>")
def course_details(course_slug):

    if course_slug not in course_map:
        return "Course not found", 404


    course_name = course_map[
        course_slug
    ]


    # Get course records
    course_df = df[
        df["Course"] == course_name
    ].copy()


    if course_df.empty:
        return "Course data not found", 404


    # Course statistics
    total_students = course_df[
        "Student_ID"
    ].nunique()


    average_marks = round(
        course_df[
            "Final_Marks"
        ].mean(),
        1
    )


    average_attendance = round(
        course_df[
            "Attendance"
        ].mean(),
        1
    )


    # Actual student records
    students = course_df.to_dict(
        orient="records"
    )


    return render_template(
        "course_details.html",

        course_name=course_name,

        total_students=total_students,

        average_marks=average_marks,

        average_attendance=average_attendance,

        students=students
    )


# =========================================================
# STUDENT DETAILS
# =========================================================

@app.route("/student/<student_id>")
def student_details(student_id):

    # Clean URL ID
    student_id = str(student_id).strip()


    # =====================================================
    # GET ALL RECORDS OF THIS STUDENT
    # =====================================================

    student_data = df[
        df["Student_ID"].astype(str).str.strip()
        == student_id
    ].copy()


    # =====================================================
    # DEBUG INFORMATION
    # =====================================================

    print("\n====================================")
    print("STUDENT ID:", student_id)
    print("RECORDS FOUND:", len(student_data))

    if not student_data.empty:

        print(
            student_data[
                [
                    "Student_ID",
                    "Student_Name",
                    "Course"
                ]
            ].to_string(index=False)
        )

    print("====================================\n")


    # Student not found
    if student_data.empty:
        return "Student not found", 404


    # =====================================================
    # STUDENT NAME
    # =====================================================

    student_name = student_data.iloc[0][
        "Student_Name"
    ]


    # =====================================================
    # SORT SUBJECTS IN FIXED ORDER
    # =====================================================

    student_data["Course"] = pd.Categorical(
        student_data["Course"],
        categories=course_order,
        ordered=True
    )


    student_data = student_data.sort_values(
        "Course"
    )


    # =====================================================
    # CONVERT ALL RECORDS TO DICTIONARY
    # =====================================================

    student_data = student_data.to_dict(
        orient="records"
    )


    # =====================================================
    # SEND DATA TO HTML
    # =====================================================

    return render_template(
        "student_details.html",

        student_id=student_id,

        student_name=student_name,

        student_data=student_data,

        course_order=course_order
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)