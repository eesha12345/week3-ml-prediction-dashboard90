
import streamlit as st
import pandas as pd
import joblib
# Load model
model = joblib.load("best_model.pkl")


st.set_page_config(page_title="Student Performance Prediction", layout="centered")

st.title("🎓 Student Performance Prediction Dashboard")

st.write("""
This application predicts the student's grade based on study hours,
attendance, class participation, and total score.
""")

st.header("Enter Student Details")

student_id = st.number_input("Student ID", min_value=1, step=1)

weekly_self_study_hours = st.number_input(
    "Weekly Self Study Hours",
    min_value=0.0,
    max_value=100.0,
)

attendance_percentage = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
)

class_participation = st.number_input(
    "Class Participation",
    min_value=0.0,
    max_value=100.0,
)

total_score = st.number_input(
    "Total Score",
    min_value=0.0,
    max_value=100.0,
)

if st.button("Predict Grade"):

    input_data = pd.DataFrame([[
        student_id,
        weekly_self_study_hours,
        attendance_percentage,
        class_participation,
        total_score
    ]], columns=[
        "student_id",
        "weekly_self_study_hours",
        "attendance_percentage",
        "class_participation",
        "total_score"
    ])

    prediction = model.predict(input_data)

    st.success(f"Predicted Grade: {prediction[0]}")

st.markdown("---")
st.subheader("Dataset Summary")

st.write("""
- Student Performance Dataset
- Machine Learning Model: Logistic Regression
- Features:
    - Student ID
    - Weekly Self Study Hours
    - Attendance Percentage
    - Class Participation
    - Total Score
- Target:
    - Grade
""")
