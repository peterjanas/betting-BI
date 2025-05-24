import streamlit as st
from Components.data_clean import data_cleaning 

st.set_page_config(page_title="Betting BI", page_icon="⚽", layout="wide")

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
    st.write('3. Is there a significant "home advantage" when comparing results over multiple seasons?')
    st.write("4. Do teams that won their last game have a higher probability of winning their current game?")
    st.write("5. Can you cluster matches into different types?")
    st.subheader("Hypotheses")
    st.write("H1: Teams with lower average betting odds (favorites) are more likely to win.")
    st.write("H2: A team ranked higher last season has a statistically significant chance of winning.")
    st.write("H3: Home teams win more often than away teams, on average.")
    st.write("H4: Teams on a winning streak have a higher chance of winning the next match.")
    st.markdown("[Link to dataset on Kaggle](https://www.kaggle.com/datasets/ivanpv/premier-league-football-matches-20152019/data?select=Premier-League-2015-2019_TRAINING.csv)")
    
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "DataCleaning"])

    if page == "Homepage":
        show_homepage()
    elif page == "DataCleaning":
        data_cleaning() 


if __name__ == "__main__":
    main()
