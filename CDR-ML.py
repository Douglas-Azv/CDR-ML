import streamlit as st
import pickle
import pandas as pd

# Load the trained Random Forest model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Function to label the GCDR scores
def lab(x):
    if x == 0.0:
        return "Health"
    elif x == 0.5:
        return "Questionable"
    elif x == 1.0:
        return "Mild"
    elif x == 2.0:
        return "Moderate"
    else:
        return "Severe"

# Streamlit Interface
st.set_page_config(
    page_title="ML-CDR Online Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ML-CDR Online Tool")
st.write("Use the sliders to input the scores and predict the CDR")

# Create columns for sliders and result
col1, col2 = st.columns(2)

# Names of the inputs
inputs = [
    "CDMEMORY",
    "CDORIENT",
    "CDJUDGE",
    "CDCOMMUN",
    "CDHOME",
    "CDCARE"
]

# Create sliders for each input with specific names
input_data = [col1.slider(name, 0.0, 3.0, step=0.1) for name in inputs]

# Convert input data to a DataFrame with correct feature names
input_df = pd.DataFrame([input_data], columns=inputs)

# Prediction and label the result
prediction = model.predict(input_df)
labeled_prediction = lab(prediction[0])
col2.write(f'CDR Prediction: {labeled_prediction}')

