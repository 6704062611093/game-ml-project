import streamlit as st
import pickle
import pandas as pd

# โหลดโมเดล
model_ml = pickle.load(open("model_ml.pkl", "rb"))
model_text = pickle.load(open("model_text.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
trained_columns = pickle.load(open("columns.pkl", "rb"))

st.title("🎮 AI Game Analytics System")

# เมนูเหลือ 2 หน้า
menu = st.sidebar.selectbox(
    "Choose Model",
    ["ML Model", "Text Model"]
)

# ======================
# 🧠 ML MODEL (รวมทุกอย่าง)
# ======================
if menu == "ML Model":
    st.header("💰 Game Spending Prediction")

    # 🔹 อธิบาย
    st.subheader("Model Description")
    st.write("""
    - Model: Stacking Classifier
    - Base Models: Random Forest, Gradient Boosting, XGBoost
    - Final Model: Logistic Regression
    - Task: Predict High Spender (0/1)
    """)

    # 🔹 input
    st.subheader("Try Prediction")

    age = st.number_input("Age", 10, 60)
    session = st.number_input("Session Count", 1, 30)
    avg_time = st.number_input("Avg Session Length", 1.0, 50.0)

    if st.button("Predict ML"):
        data = pd.DataFrame([{
            "Age": age,
            "SessionCount": session,
            "AverageSessionLength": avg_time,
            "Gender": "Male",
            "Country": "USA",
            "Device": "Android",
            "GameGenre": "Action",
            "SpendingSegment": "Minnow"
        }])

        data = pd.get_dummies(data)
        data = data.reindex(columns=trained_columns, fill_value=0)

        pred = model_ml.predict(data)

        if pred[0] == 1:
            st.success("High Spender 💰")
        else:
            st.warning("Low Spender 🪙")

# ======================
# 💬 TEXT MODEL (รวมทุกอย่าง)
# ======================
elif menu == "Text Model":
    st.header("📝 Steam Review Analysis")

    # 🔹 อธิบาย
    st.subheader("Model Description")
    st.write("""
    - Model: LinearSVC
    - Technique: TF-IDF
    - Task: Sentiment Classification
    - Output: Positive / Not Positive
    """)

    # 🔹 input
    st.subheader("Try Prediction")

    text = st.text_area("Enter game description")

    if st.button("Predict Text"):
        if text.strip() == "":
            st.warning("Please enter text")
        else:
            data = vectorizer.transform([text])
            pred = model_text.predict(data)

            st.success(f"Prediction: {pred[0]}")