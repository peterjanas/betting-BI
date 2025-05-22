import streamlit as st
from Components.data_clean import data_cleaning 

st.set_page_config(page_title="Betting BI", page_icon="⚽", layout="wide")

def show_homepage():
    st.title("BETTING DATA DASHBOARD")
    st.write("Welcome to our football betting data analysis project, which focuses on predicting outcomes based on various metrics")
    st.write("Made by Peter, Chris, Masih, Umair and Tobias")
    
    
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "DataCleaning"])

    if page == "Homepage":
        show_homepage()
    elif page == "DataCleaning":
        data_cleaning() 


if __name__ == "__main__":
    main()
