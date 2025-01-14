import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('lookalike_model.pkl', 'rb'))

# Streamlit UI
st.title("Lookalike Detection App")

# File uploader
uploaded_file = st.file_uploader("Upload your leads file (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Leads Data:", data.head())

    # Use the model to predict lookalikes
    distances, indices = model.kneighbors(data)
    st.write("Lookalike prediction completed!")
    st.write("Nearest Neighbors (Indices):", indices)
