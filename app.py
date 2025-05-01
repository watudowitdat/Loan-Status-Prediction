import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and preprocessing tools
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# Get category options from encoder
gender_options = encoder.categories_[0].tolist()
education_options = encoder.categories_[1].tolist()
ownership_options = encoder.categories_[2].tolist()
intent_options = encoder.categories_[3].tolist()
default_history_options = encoder.categories_[4].tolist()

st.title("Loan Default Risk Predictor")
st.write("Please enter applicant information to predict the probability of loan **non-default**.")

with st.form("loan_form"):
    person_gender = st.selectbox("Gender", gender_options)
    person_education = st.selectbox("Education", education_options)
    person_home_ownership = st.selectbox("Home Ownership", ownership_options)
    loan_intent = st.selectbox("Loan Purpose", intent_options)
    previous_loan_defaults_on_file = st.selectbox("Previous Default on File?", default_history_options)
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=15000)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=10.0)
    loan_percent_income = st.number_input("Loan-to-Income Ratio (%)", min_value=0.0, max_value=100.0, value=25.0)
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=680)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
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

    # Transform categorical and numeric features
    cat_vars = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]
    num_vars = ["loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]

    encoded_cat = encoder.transform(input_df[cat_vars])
    scaled_num = scaler.transform(input_df[num_vars])

    df_cat = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_vars))
    df_num = pd.DataFrame(scaled_num, columns=num_vars)

    X_input = pd.concat([df_cat, df_num], axis=1)

    y_prob = model.predict(X_input)[0][0]
    y_binary = int(y_prob > 0.65)

    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.write(f"**Probability of NOT Defaulting**: `{y_prob:.2%}`")
    st.write(f"**Final Classification**: {'✅ Low Risk (1)' if y_binary == 1 else '⚠️ High Risk (0)'}")
