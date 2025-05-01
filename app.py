import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load the model and with scaler and encoder for data preprocess
try:
    model = load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
except Exception as e:
    st.error("❌ 模型或预处理文件加载失败，请检查文件是否上传正确。")
    st.exception(e)
    st.stop()

gender_options = ["male", "female"]
education_options = ["High School", "College", "Graduate", "Other"]
ownership_options = ["rent", "own", "mortgage", "other"]
intent_options = ["personal", "education", "medical", "venture", "home"]
default_history_options = ["yes", "no"]

st.title("Loan status prediction")
st.write("Enter applicant's information below to generate prediction.")

# -----------------------------
# Sections
with st.form("loan_form"):
    person_gender = st.selectbox("Gender", gender_options)
    person_education = st.selectbox("Education", education_options)
    person_home_ownership = st.selectbox("Homeownership", ownership_options)
    loan_intent = st.selectbox("Intention for loan", intent_options)
    previous_loan_defaults_on_file = st.selectbox("Default record (yes/no)", default_history_options)
    loan_amnt = st.number_input("Loan amount", min_value=500, max_value=50000, value=15000)
    loan_int_rate = st.number_input("Interest rate (%)", min_value=0.0, max_value=40.0, value=12.5)
    loan_percent_income = st.number_input("Loan to income ratio (%)", min_value=0.0, max_value=100.0, value=25.0)
    cb_person_cred_hist_length = st.number_input("Length of credit history (year)", min_value=0, max_value=50, value=10)
    credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=650)

    submitted = st.form_submit_button("Predicted rate of not defaulting")

# -----------------------------
# Execute prediction
if submitted:
    # Generate DataFrame
    input_dict = {
        "person_gender": [person_gender],
        "person_education": [person_education],
        "person_home_ownership": [person_home_ownership],
        "loan_intent": [loan_intent],
        "previous_loan_defaults_on_file": [previous_loan_defaults_on_file],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
        "credit_score": [credit_score],
    }

    input_df = pd.DataFrame(input_dict)

 #  OneHot Encoded Categorical Variable（Use trained encoder）
    cat_var = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]
    cat_encoded = encoder.transform(input_df[cat_var])
    cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_var))

#   Scaler for numerical features（Use trained scaler）
    num_var = ["loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]
    scaled_num = scaler.transform(input_df[num_var])
    num_df = pd.DataFrame(scaled_num, columns=num_var)

    # Integrate the inputs
    full_input = pd.concat([cat_df, num_df], axis=1)

    # Model prediction
    y_pred_prob = model.predict(full_input)[0][0]
    y_pred_binary = int(y_pred_prob > 0.65)

    # Return Result
    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.write(f"📊 Predicted Default rate：**{y_pred_prob:.2%}**")
    st.write(f"🏷️ Classification Result：**{'High Default Risk (0)' if y_pred_binary == 0 else 'Low Default Risk (1)'}**")
