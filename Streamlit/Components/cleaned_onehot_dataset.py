import streamlit as st
import pandas as pd

def load_cleaned_onehot_data():
    st.title("Cleaned One Hot Encoding Premier League Dataset")
    clean_onehot_dataset = pd.read_csv('../data/cleaned-premier-onehot.csv')
    st.dataframe(clean_onehot_dataset)

    