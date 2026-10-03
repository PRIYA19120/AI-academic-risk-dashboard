import pandas as pd
import random

# =========================================================
# SETTINGS
# =========================================================

random.seed(42)

courses = [
    "Machine Learning",
    "DBMS",
    "Cloud Computing",
    "Data Structures",
    "Python",
    "Data Science"
]


first_names = [
    "Aarav", "Aditi", "Ananya", "Arjun", "Diya",
    "Ishita", "Kavya", "Karan", "Meera", "Nikhil",
    "Priya", "Rahul", "Riya", "Rohan", "Simran",
    "Sneha", "Tanvi", "Varun", "Vansh", "Yash"
]


last_names = [
    "Sharma", "Joshi", "Verma", "Singh", "Gupta",
    "Kumar", "Rawat", "Negi", "Mehta", "Bisht"
]


# =========================================================
# CREATE 30 UNIQUE STUDENTS
# =========================================================

students = []

used_names = set()

while len(students) < 30:

    first = random.choice(first_names)
    last = random.choice(last_names)

    name = first + " " + last

    if name not in used_names:

        used_names.add(name)

        students.append(name)


# =========================================================
# GENERATE ACADEMIC DATA
# =========================================================

data = []


# IMPORTANT:
# Same student gets all 6 courses

for student_number, student_name in enumerate(
    students,
    start=1
):

    student_id = f"S{student_number:03d}"


    for course in courses:

        attendance = random.randint(
            50, 98
        )

        assignment = random.randint(
            40, 95
        )

        quiz = random.randint(
            40, 95
        )

        internal = random.randint(
            40, 95
        )

        exam = random.randint(
            40, 95
        )


        # Final marks calculation

        final_marks = round(
            (
                assignment * 0.15
                + quiz * 0.15
                + internal * 0.25
                + exam * 0.45
            ),
            2
        )


        data.append([

            student_id,

            student_name,

            course,

            attendance,

            assignment,

            quiz,

            internal,

            exam,

            final_marks
        ])


# =========================================================
# CREATE DATAFRAME
# =========================================================

columns = [

    "Student_ID",

    "Student_Name",

    "Course",

    "Attendance",

    "Assignment",

    "Quiz",

    "Internal",

    "Exam",

    "Final_Marks"
]


df = pd.DataFrame(
    data,
    columns=columns
)


# =========================================================
# SAVE DATASET
# =========================================================

df.to_csv(
    "dataset/academic_data.csv",
    index=False
)


# =========================================================
# CHECK DATASET
# =========================================================

print("===================================")
print("Dataset generated successfully!")
print("===================================")

print(
    "Unique Students:",
    df["Student_ID"].nunique()
)

print(
    "Total Records:",
    len(df)
)

print(
    "Total Courses:",
    df["Course"].nunique()
)

print()
print("Records per student:")

print(
    df.groupby("Student_ID")
      .size()
      .head()
)

print()
print("Sample data:")

print(
    df.head(12)
)

print()
print(
    "File: dataset/academic_data.csv"
)