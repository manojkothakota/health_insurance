# ============================================
# INSURANCE PREDICTION STREAMLIT APP
# ============================================

import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ============================================
# TITLE
# ============================================

st.title("💡 Insurance Charges Prediction")
st.write("Enter the details below 👇")

# ============================================
# MODEL TRAINING (CACHED)
# ============================================

@st.cache_resource
def train_model():
    df = pd.read_csv(r'D:\sem-4\ml\health_insurance\insurance.csv')

    # Encoding
    df['sex'] = df['sex'].map({'male':0, 'female':1})
    df['smoker'] = df['smoker'].map({'no':0, 'yes':1})
    df = pd.get_dummies(df, columns=['region'], drop_first=True)

    # Feature engineering
    df["bmi_smoker"] = df["bmi"] * df["smoker"]

    # Log transform
    df['charges'] = np.log(df['charges'])

    # Split
    X = df.drop('charges', axis=1)
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(0.001), loss='mse')

    model.fit(X_train, y_train, epochs=50, verbose=0)

    return model, scaler, X.columns

# Load model
model, scaler, columns = train_model()

# ============================================
# USER INPUT
# ============================================

age = st.slider("Age", 18, 65, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northwest", "southeast", "southwest", "northeast"])

# ============================================
# ENCODE INPUT
# ============================================

sex = 0 if sex == "male" else 1
smoker_val = 1 if smoker == "yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

bmi_smoker = bmi * smoker_val

# Create input dataframe (IMPORTANT)
input_dict = {
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker_val,
    'region_northwest': region_northwest,
    'region_southeast': region_southeast,
    'region_southwest': region_southwest,
    'bmi_smoker': bmi_smoker
}

input_df = pd.DataFrame([input_dict])

# Ensure column order matches training
input_df = input_df.reindex(columns=columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# ============================================
# PREDICTION
# ============================================

if st.button("Predict Charges"):
    prediction = model.predict(input_scaled)

    # Safe conversion
    prediction = np.clip(prediction, -10, 10)
    final = np.exp(prediction)[0][0]

    st.success(f"💰 Predicted Insurance Charges: ₹ {round(final, 2)}")