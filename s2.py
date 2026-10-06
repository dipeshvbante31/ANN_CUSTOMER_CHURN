import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# THEME
# =========================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

with st.sidebar:
    st.title("⚙️ Settings")

    dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.dark_mode
    )

    st.session_state.dark_mode = dark_mode

    st.divider()

    st.markdown("### About")
    st.write(
        "This application uses a trained deep learning regression "
        "model to estimate a customer's annual salary."
    )


# =========================================================
# LOAD MODEL AND PREPROCESSORS
# =========================================================

@st.cache_resource
def load_resources():

    model = tf.keras.models.load_model("regression.h5")

    with open("label_gen.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("geo.pkl", "rb") as file:
        ohe_encoder_geo = pickle.load(file)

    with open("stdscaler2.pkl", "rb") as file:
        scaler = pickle.load(file)

    return (
        model,
        label_encoder_gender,
        ohe_encoder_geo,
        scaler
    )


try:
    (
        model,
        label_encoder_gender,
        ohe_encoder_geo,
        scaler
    ) = load_resources()

except Exception as e:
    st.error("Unable to load the trained model or preprocessing files.")
    st.exception(e)
    st.stop()


# =========================================================
# THEME CSS
# =========================================================

if dark_mode:

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #0e1117;
        }

        h1, h2, h3 {
            color: #ffffff !important;
        }

        p, label {
            color: #d1d5db !important;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff !important;
        }

        [data-testid="stMetricLabel"] {
            color: #d1d5db !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #ffffff;
        }

        h1, h2, h3 {
            color: #111827 !important;
        }

        p, label {
            color: #374151 !important;
        }

        [data-testid="stMetricValue"] {
            color: #111827 !important;
        }

        [data-testid="stMetricLabel"] {
            color: #374151 !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HEADER
# =========================================================

st.title("💰 Estimated Salary Prediction")

st.caption(
    "Enter customer information to estimate their annual salary "
    "using a trained deep learning regression model."
)

st.divider()


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("👤 Customer Information")


# ---------------------------------------------------------
# PERSONAL DETAILS
# ---------------------------------------------------------

st.markdown("### Personal Details")

col1, col2, col3 = st.columns(3)

with col1:

    geography = st.selectbox(
        "🌍 Geography",
        options=ohe_encoder_geo.categories_[0]
    )

with col2:

    gender = st.selectbox(
        "👤 Gender",
        options=label_encoder_gender.classes_
    )

with col3:

    age = st.slider(
        "🎂 Age",
        min_value=18,
        max_value=92,
        value=35
    )


# ---------------------------------------------------------
# FINANCIAL DETAILS
# ---------------------------------------------------------

st.markdown("### 💳 Financial Details")

col1, col2, col3 = st.columns(3)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1
    )

with col2:

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        format="%.2f"
    )

with col3:

    tenure = st.slider(
        "Tenure",
        min_value=0,
        max_value=19,
        value=5
    )


# ---------------------------------------------------------
# ACCOUNT DETAILS
# ---------------------------------------------------------

st.markdown("### 🏦 Account Details")

col1, col2, col3, col4 = st.columns(4)

with col1:

    num_of_products = st.selectbox(
        "Number of Products",
        options=[1, 2, 3, 4],
        index=0
    )

with col2:

    has_cr_card = st.selectbox(
        "Has Credit Card",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col3:

    is_active_member = st.selectbox(
        "Active Member",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col4:

    exited = st.selectbox(
        "Exited",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "🔮 Predict Estimated Salary",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # -------------------------------------------------
        # ENCODE GENDER
        # -------------------------------------------------

        gender_encoded = label_encoder_gender.transform(
            [gender]
        )[0]


        # -------------------------------------------------
        # CREATE NUMERICAL INPUT DATA
        # -------------------------------------------------

        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [gender_encoded],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "Exited": [exited]
        })


        # -------------------------------------------------
        # ENCODE GEOGRAPHY
        # -------------------------------------------------

        geo_input = pd.DataFrame({
            "Geography": [geography]
        })

        geo_encoded = ohe_encoder_geo.transform(
            geo_input
        )

        # Handle sparse/dense encoder output
        if hasattr(geo_encoded, "toarray"):
            geo_encoded = geo_encoded.toarray()

        geo_feature_names = (
            ohe_encoder_geo
            .get_feature_names_out(["Geography"])
        )

        # Check encoder output
        if geo_encoded.shape[1] != len(geo_feature_names):

            st.error(
                "The Geography encoder does not match the "
                "saved model preprocessing configuration."
            )

            st.write(
                "Encoder output shape:",
                geo_encoded.shape
            )

            st.write(
                "Expected Geography features:",
                len(geo_feature_names)
            )

            st.stop()


        geo_df = pd.DataFrame(
            geo_encoded,
            columns=geo_feature_names
        )


        # -------------------------------------------------
        # COMBINE ALL FEATURES
        # -------------------------------------------------

        input_data = pd.concat(
            [
                input_data.reset_index(drop=True),
                geo_df.reset_index(drop=True)
            ],
            axis=1
        )


        # -------------------------------------------------
        # SCALE INPUT
        # -------------------------------------------------

        input_data_scaled = scaler.transform(
            input_data
        )


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            input_data_scaled,
            verbose=0
        )

        prediction_salary = float(
            prediction[0][0]
        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.subheader("📊 Prediction Result")

        st.success(
            f"Estimated Annual Salary: ${prediction_salary:,.2f}"
        )


        # -------------------------------------------------
        # SALARY METRICS
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="Annual Salary",
                value=f"${prediction_salary:,.2f}"
            )

        with col2:

            monthly_salary = prediction_salary / 12

            st.metric(
                label="Estimated Monthly Salary",
                value=f"${monthly_salary:,.2f}"
            )


        # -------------------------------------------------
        # INPUT SUMMARY
        # -------------------------------------------------

        st.markdown("### 📋 Customer Summary")

        summary = pd.DataFrame({
            "Feature": [
                "Geography",
                "Gender",
                "Age",
                "Credit Score",
                "Balance",
                "Tenure",
                "Number of Products",
                "Has Credit Card",
                "Active Member",
                "Exited"
            ],

            "Value": [
                geography,
                gender,
                age,
                credit_score,
                f"${balance:,.2f}",
                tenure,
                num_of_products,
                "Yes" if has_cr_card == 1 else "No",
                "Yes" if is_active_member == 1 else "No",
                "Yes" if exited == 1 else "No"
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


        # -------------------------------------------------
        # MODEL INPUT
        # -------------------------------------------------

        with st.expander("🔍 View Model Input"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "Something went wrong while generating the prediction."
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "💡 Salary Prediction System • Deep Learning Regression Model"
)