import streamlit as st
import pandas as pd

def load_old_data():
    st.title("Uncleaned Premier League Dataset")
    old_dataset = pd.read_csv('../data/Premier-League-2015-2019.csv')
    st.dataframe(old_dataset)

    