import streamlit as st 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle

model = tf.keras.models.load_model('regression.h5')


with open("label_gen.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("geo.pkl", "rb") as file:
    ohe_encoder_geo = pickle.load(file)

with open("stdscaler2.pkl", "rb") as file:
    scaler = pickle.load(file)

# streamlit app

st.title('Estimated salary prediction')

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

exited = st.selectbox('Exited',[0,1])

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

#prepare input data 
input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "Exited":[exited]
    })

geo_encoded=ohe_encoder_geo.transform([[geography]]).toarray()
geo_df=pd.DataFrame(geo_encoded,columns=ohe_encoder_geo.get_feature_names_out(['Geography']))

##combine ohe encoded columns with input data 
input_data=pd.concat([input_data.reset_index(drop=True),geo_df])

input_data_scaled= scaler.transform(input_data)

prediction = model.predict( input_data_scaled)
prediction_salary=prediction[0][0]
st.write(f'predcted estimated salary:${prediction_salary:.2f}')
