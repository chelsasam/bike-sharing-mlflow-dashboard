
import streamlit as st
import pandas as pd
import os
import json

st.set_page_config(
    page_title="Bike Sharing Demand Prediction",
    page_icon="🚲",
    layout="wide"
)

ROOT = os.path.dirname(os.path.abspath(__file__))

st.title("Bike Sharing Demand Prediction")
st.subheader("Machine Learning Model Performance Dashboard")

st.write(
    "This dashboard compares four machine learning models "
    "trained to predict bike-sharing demand."
)

# Load actual model comparison results
file_path = os.path.join(
    ROOT, "reports", "final_model_comparison.csv"
)

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    st.header("Model Comparison")
    st.dataframe(df, use_container_width=True)

    st.header("Model Performance")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        selected_metric = st.selectbox(
            "Select performance metric",
            numeric_columns
        )

        st.bar_chart(
            df.set_index(df.columns[0])[[selected_metric]]
        )
else:
    st.error("Model comparison file not found.")

# Load API prediction result
prediction_file = os.path.join(
    ROOT, "reports", "api_prediction_result.json"
)

if os.path.exists(prediction_file):
    with open(prediction_file, "r") as f:
        prediction = json.load(f)

    st.header("API Prediction Result")
    st.json(prediction)

st.success("Dashboard loaded successfully!")
