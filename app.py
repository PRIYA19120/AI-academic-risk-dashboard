from flask import Flask, render_template
import pandas as pd
app = Flask(__name__)
df = pd.read_csv("dataset/academic_data.csv")


@app.route("/")
def dashboard():

    total_students = df["Student_ID"].nunique()

    average_attendance = round(df["Attendance"].mean(), 1)

    average_marks = round(df["Final_Marks"].mean(), 1)

    courses_count = df["Course"].nunique()

    return render_template(
        "index.html",
        total_students=total_students,
        average_attendance=average_attendance,
        average_marks=average_marks,
        courses_count=courses_count
    )

@app.route("/courses")
def courses():
    return render_template("courses.html")
@app.route("/course/machine-learning")
def machine_learning():
    return render_template(
        "course_details.html",
        course_name="Machine Learning" )
@app.route("/course/dbms")
def dbms():
    return render_template("course_details.html", course_name="DBMS")


@app.route("/course/cloud-computing")
def cloud_computing():
    return render_template("course_details.html", course_name="Cloud Computing")


@app.route("/course/data-structures")
def data_structures():
    return render_template("course_details.html", course_name="Data Structures")


@app.route("/course/python")
def python():
    return render_template("course_details.html", course_name="Python")


@app.route("/course/data-science")
def data_science():
    return render_template("course_details.html", course_name="Data Science")


if __name__ == "__main__":
    app.run(debug=True)