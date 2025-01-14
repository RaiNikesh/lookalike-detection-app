import streamlit as st
import pandas as pd
import pickle

# Load the model
model = pickle.load(open('lookalike_model.pkl', 'rb'))

# Streamlit App
st.title('Lookalike Prediction App')

uploaded_file = st.file_uploader('Upload your dataset (Excel format)', type=['xlsx'])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write('Dataset preview:')
    st.write(df.head())

    # Make predictions
    predictions = model.predict(df)
    df['Lookalike_Predictions'] = predictions
    st.write('Predictions:')
    st.write(df)

    # Download link for predictions
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name='lookalike_predictions.csv',
        mime='text/csv',
    )
