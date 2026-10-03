
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import streamlit as st


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = tf.keras.models.load_model("model.h5")


# =========================================================
# LOAD SCALER AND ENCODERS
# =========================================================

with open("ohe_gen.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("ohe_geo.pkl", "rb") as file:
    ohe_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# =========================================================
# STREAMLIT APP
# =========================================================

st.title("Customer Churn Prediction")
st.write("Enter the customer's details to predict the probability of churn.")


# =========================================================
# USER INPUT
# =========================================================

geography = st.selectbox(
    "Geography",
    ohe_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=92,
    value=35
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=0.0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=19,
    value=5
)

num_of_products = st.slider(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)


# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Churn"):

    # Encode Gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Create numerical input DataFrame
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })


    # =====================================================
    # ONE-HOT ENCODE GEOGRAPHY
    # =====================================================

    # transform() is already returning a NumPy array,
    # therefore .toarray() is NOT required.
    geo_encoded = ohe_encoder_geo.transform([[geography]])

    geo_df = pd.DataFrame(
        geo_encoded,
        columns=ohe_encoder_geo.get_feature_names_out(["Geography"])
    )


    # =====================================================
    # COMBINE FEATURES
    # =====================================================

    input_data = pd.concat(
        [
            input_data.reset_index(drop=True),
            geo_df.reset_index(drop=True)
        ],
        axis=1
    )


    # =====================================================
    # SCALE INPUT
    # =====================================================

    input_data_scaled = scaler.transform(input_data)


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(input_data_scaled, verbose=0)

    prediction_proba = float(prediction[0][0])


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.subheader("Prediction Result")

    st.write(
        f"Churn Probability: **{prediction_proba:.2%}**"
    )

    if prediction_proba > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is unlikely to churn.")
