import pandas as pd
import random

# -----------------------------
# SETTINGS
# -----------------------------

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


# -----------------------------
# GENERATE DATA
# -----------------------------

data = []

student_id = 1

for course in courses:

    for i in range(30):

        student_name = (
            random.choice(first_names)
            + " "
            + random.choice(last_names)
        )

        attendance = random.randint(50, 98)
        assignment = random.randint(40, 95)
        quiz = random.randint(40, 95)
        internal = random.randint(40, 95)
        exam = random.randint(40, 95)

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
            f"S{student_id:03d}",
            student_name,
            course,
            attendance,
            assignment,
            quiz,
            internal,
            exam,
            final_marks
        ])

        student_id += 1


# -----------------------------
# CREATE DATAFRAME
# -----------------------------

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

df = pd.DataFrame(data, columns=columns)


# -----------------------------
# SAVE CSV
# -----------------------------

df.to_csv(
    "dataset/academic_data.csv",
    index=False
)

print("===================================")
print("Dataset generated successfully!")
print("===================================")
print("Total Students:", len(df))
print("Total Courses:", len(courses))
print("File: dataset/academic_data.csv")
print()
print(df.head())