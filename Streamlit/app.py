import streamlit as st

from Components.naive_bayes_page import show_nb_model_page
from Components.random_forest_page import show_randomforest_model_page
st.set_page_config(page_title="Betting BI", page_icon="⚽", layout="wide")  # 🔺 Must be FIRST

from Components.naive_bayes_page import show_nb_model_page
from Components.old_dataset import load_old_data
from Components.cleaned_label_dataset import load_cleaned_label_data
from Components.cleaned_onehot_dataset import load_cleaned_onehot_data
from Components.eda import load_eda


def show_homepage():
    st.title("BETTING DATA DASHBOARD")
    st.write("Welcome to our football betting data analysis project, which focuses on predicting outcomes based on various metrics")
    st.write("Made by Peter, Chris, Masih, Umair and Tobias")
    st.subheader("Problem statement")
    st.write("Predicting the outcome of football matches is inherently difficult due to the unpredictable nature of the game and the many variables that can influence results. This poses a challenge, especially as bookmakers set odds strategically to maximize their own advantage. To address this, we aim to apply machine learning techniques to analyze historical match data and uncover patterns or variables that may influence outcomes. By doing so, we hope to better understand the factors that affect match results and potentially improve prediction accuracy and potentially help individuals make more informed decisions when interpreting them.")
    st.subheader("Project annotation")
    st.write("We aim to explore how various factors influence the outcome of football matches, such as betting odds, previous rankings, and recent performance. The challenge lies in uncovering whether consistent patterns exist that can help predict match results more accurately. Our project will build a data-driven model using historical match data and betting information to identify key predictors of match outcomes. This solution could benefit sports analysts, betting companies, and fans by providing insights into game trends and improving forecasting accuracy.")
    st.subheader("Context and purpose")
    st.write("Predicting the outcome of football matches is a complex task influenced by numerous variables, from team form and home advantage to odds provided by bookmakers. With the increasing availability of sports and betting data, this project aims to apply Business Intelligence (BI) techniques to discover meaningful patterns and relationships. The purpose is to assist decision-making in sports analytics, provide insights into match dynamics, and explore how external data (like odds) may reflect or predict actual performance.")
    st.subheader("Research questions")
    st.write("1. How do bookmaker odds correlate with actual match outcomes?")
    st.write("2. Does a higher ranking from the previous season increase the likelihood of winning?")
    st.write('3. Does the home team have an advantage over the away team?')
    st.write("4. Do teams that won their last game have a higher probability of winning their current game?")
    st.write("5. Can we accurately predict the outcome of a match using our data, and which machine learning model performs best for this task?")
    st.subheader("Hypotheses")
    st.write("H1: Teams with lower average betting odds (favorites) are more likely to win.")
    st.write("H2: A team ranked higher last season has a statistically significant chance of winning.")
    st.write("H3: Home teams win more often than away teams, on average.")
    st.write("H4: Teams on a winning streak have a higher chance of winning the next match.")
    st.write("H5: We believe match outcomes can be predicted with reasonable accuracy, and that a classification model such as a decision tree would be the most accurate.")
    st.markdown("[Link to dataset on Kaggle](https://www.kaggle.com/datasets/ivanpv/premier-league-football-matches-20152019/data?select=Premier-League-2015-2019_TRAINING.csv)")
    
def main():
    st.sidebar.title("Navigation")
    
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Original-Dataset","Label-Dataset", "One-Hot-Dataset", "EDA", "NB-Model","Random-Forest-Model"])

    if page == "Homepage":
        show_homepage()
    elif page == "Original-Dataset":
        load_old_data()
    elif page == "Label-Dataset":
        load_cleaned_label_data()
    elif page == "One-Hot-Dataset":
        load_cleaned_onehot_data()
    elif page == "EDA":
        load_eda()
    elif page == "NB-Model":
        show_nb_model_page() 
    elif page == "Random-Forest-Model":
        show_randomforest_model_page()
        
                       



if __name__ == "__main__":
    main()
