import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Page Configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Title
st.title("🌸 Iris Flower Classification")

st.write(
    """
    This application predicts the **species of an Iris flower**
    using a Machine Learning model trained with the Iris dataset.
    Enter the flower measurements below and click **Predict**.
    """
)

# Sidebar
st.sidebar.header("About")
st.sidebar.write(
    """
    **Algorithms Compared**
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Random Forest

    **Final Model:** Random Forest Classifier
    """
)

# User Inputs
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

# Prediction
if st.button("🔮 Predict"):

    import pandas as pd

    input_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    flower_names = [
        "Setosa 🌼",
        "Versicolor 🌷",
        "Virginica 🌺"
    ]

    st.success(
        f"Predicted Species: **{flower_names[prediction[0]]}**"
    )
    
    st.divider()

    st.info(
        """
        This model was trained on the Iris dataset using
        a Random Forest Classifier and compares multiple
        machine learning algorithms for classification.
        """
    )
    
    st.caption("Developed by Anushka Soni")