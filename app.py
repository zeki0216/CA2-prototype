import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Page title
st.title("HK Used Car Price Estimator")

# Load dataset
df = pd.read_csv("Lab04_hk_car_price.csv")

# Features used in the improved model
FEATURES = [
    "Manufacture_Year",
    "Mileage_km",
    "Brand",
    "Horsepower_PS",
    "Car_Age_At_Sale",
    "Displacement_cc"
]

# Target column
TARGET = "Price_HKD"

# Prepare training data
X_all = df[FEATURES]
y_all = df[TARGET]

# Numeric columns
numeric_features = [
    "Manufacture_Year",
    "Mileage_km",
    "Horsepower_PS",
    "Car_Age_At_Sale",
    "Displacement_cc"
]

# Categorical columns
categorical_features = ["Brand"]

# Handle numeric missing values
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Handle categorical missing values and encode text
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Train model
model.fit(X_all, y_all)

# Input section
st.header("Enter Vehicle Information")

brand = st.selectbox(
    "Brand",
    sorted(df["Brand"].dropna().unique())
)

manufacture_year = st.number_input(
    "Manufacture Year",
    min_value=2000,
    max_value=2026,
    value=2020
)

mileage = st.number_input(
    "Mileage (km)",
    min_value=0,
    value=50000
)

horsepower = st.number_input(
    "Horsepower (PS)",
    min_value=1,
    value=150
)

car_age = st.number_input(
    "Car Age At Sale",
    min_value=0,
    value=5
)

displacement = st.number_input(
    "Displacement (cc)",
    min_value=0,
    value=2000
)

# Predict button
if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "Manufacture_Year": [manufacture_year],
        "Mileage_km": [mileage],
        "Brand": [brand],
        "Horsepower_PS": [horsepower],
        "Car_Age_At_Sale": [car_age],
        "Displacement_cc": [displacement]
    })

    predicted_price = model.predict(input_df)[0]

    st.success(f"Estimated Resale Price: HK${predicted_price:,.0f}")