import streamlit as st
import pandas as pd

def load_cleaned_label_data():
    st.title("Cleaned Label Encoding Premier League Dataset")
    clean_label_dataset = pd.read_csv('../data/cleaned-premier-label.csv')
    st.dataframe(clean_label_dataset)

    