import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('heart.csv')

# Prepare data
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Heart Disease Prediction")

# Collect user input
age = st.number_input("Age")
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol Level")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina", [0, 1])

# Prediction button
if st.button("Predict"):
    user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang]]
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    
    if prediction == 1:
        st.write("The patient is likely to have heart disease.")
    else:
        st.write("The patient is unlikely to have heart disease.")
