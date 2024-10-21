import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st
import pickle

# Carregar os dados reais
df = pd.read_csv("Downloads/df_cdr_final.csv")
data = df

# Define independent and dependent variables
X = df.drop(columns=['GCDR'])
y = data['GCDR']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Salvar o modelo treinado
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

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
prediction = rf.predict(input_df)
labeled_prediction = lab(prediction[0])
col2.write(f'CDR Prediction: {labeled_prediction}')
