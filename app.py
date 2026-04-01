import streamlit as st
import pickle
import pandas as pd

# โหลดโมเดล
model_ml = pickle.load(open("model_ml.pkl", "rb"))
model_text = pickle.load(open("model_text.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
trained_columns = pickle.load(open("columns.pkl", "rb"))

st.title("🎮 AI Game Analytics System")

menu = st.sidebar.selectbox(
    "Menu",
    [
        "ML Model Description",
        "ML Prediction",
        "Text Model Description",
        "Text Prediction"
    ]
)

# ======================
# 📊 ML DESCRIPTION
# ======================
if menu == "ML Model Description":
    st.header("💰 Machine Learning Model")

    st.write("""
### 1. Data Preparation
- Dataset: Mobile Game In-App Purchases
- Removed UserID
- Handled missing values (mean)
- One-hot encoding
- Created target (HighSpender)

### 2. Algorithm Theory
- Random Forest (bagging)
- Gradient Boosting (boosting)
- XGBoost (optimized boosting)
- Stacking (combine models)

### 3. Model Development
- Split data 80/20
- Train multiple models
- Combine using stacking
- Evaluate with accuracy

### 4. Data Source
- Kaggle Dataset
""")

# ======================
# 🤖 ML PREDICTION
# ======================
elif menu == "ML Prediction":
    st.header("🎮 Predict Game Spending")

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
# 🧠 TEXT DESCRIPTION
# ======================
elif menu == "Text Model Description":
    st.header("📝 Text Classification Model")

    st.write("""
### 1. Data Preparation
- Dataset: Steam Reviews
- Removed missing values
- Used description as input
- TF-IDF transformation

### 2. Algorithm Theory
- TF-IDF: text → numeric
- LinearSVC: classification using hyperplane

### 3. Model Development
- Split data 80/20
- Train SVM model
- Evaluate with accuracy

### 4. Data Source
- Kaggle Dataset
""")

# ======================
# 💬 TEXT PREDICTION
# ======================
elif menu == "Text Prediction":
    st.header("🎮 Steam Review Predictor")

    text = st.text_area("Enter game description")

    if st.button("Predict Text"):
        if text.strip() == "":
            st.warning("Please enter text")
        else:
            data = vectorizer.transform([text])
            pred = model_text.predict(data)

            st.success(f"Prediction: {pred[0]}")