# 🩺 Diabetes Prediction Web App

This is a Flask-based web application that predicts whether a person is likely to have diabetes using a machine learning model trained on medical data.

## 📊 Project Overview

- A logistic regression model is trained to predict diabetes.
- The user inputs health metrics like BMI, blood pressure, insulin level, etc.
- The model processes the data and returns a prediction: **Diabetic** or **Non-Diabetic**.
- A simple and user-friendly interface built with Flask.

## 🧠 Machine Learning Model

- Trained using the [PIMA Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
- Features used: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
- Model: Logistic Regression (can be extended to other models like Random Forest, XGBoost, etc.)

## 🗂 Folder Structure

diabetes-prediction-app/
│
├── model/ 
│ ├── trained_model.pkl
│ └── scaler.pkl
│
├── templates/ 
│ ├── index.html
│ └── results.html
│
├── static/
│ └── style.css
│
├── dataset/ 
│
├── plots/
│
├── app.py 
├── model_training.py 
├── requirements.txt 
└── README.md 
