import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def load_exam():


    st.title("BI project: premier league and betting odds")

  
    df = pd.read_csv('../data/cleaned-premier-label.csv')
    
    st.subheader("What we wanted to research")
    st.write("Our dataset contains information about Premier League matches from the 2015-2019 seasons, including team statistics, match outcomes, and betting odds.")
    
    
    st.subheader("")
    
    
    
    st.subheader("Error in the dataset")
    
    
    st.image("../Pics/match-outcome-distribution.png", caption="", width=1000)


    outcome_data = {
    "Home Win": 43.74,
    "Draw": 25.84,
    "Away Win": 30.42
}
    st.subheader("Outcome Distribution")

    col1, col2 = st.columns([1, 2])

    with col1:
        for outcome, percent in outcome_data.items():
            st.write(f"**{outcome}**: {percent:.2f}%")

    st.subheader("")
    
    
    st.subheader("Histogram for Home and Away Wins")

    col1, col2 = st.columns(2)

    with col1:
        st.image("../Pics/histogram-home-win.png", caption="", width=700)

    with col2:
        st.image("../Pics/histogram-away-win.png", caption="", width=700)
        
    st.subheader("")

        
    
    
    st.subheader("Average odds by each outcome")

    st.image("../Pics/averge-odds-by-outcome.png", caption="", width=1200)
    
    st.subheader("")
    
    
    
    st.subheader("how often were each outcome favored by the odds?")
    
    st.image("../Pics/odds-outcome-favored.png", caption="", width=900)
    
    st.subheader("")
    
  
  
    
    
    st.subheader("Is the home odds actually correct?")
    
    st.image("../Pics/correlation-between-odds-and-outcome.png", caption="", width=1000)
    
    st.subheader("What could it be and what it means for our models")
    st.write("1. ")
    st.write("2. ")
    st.write("3. ")
    
    st.subheader("")
    
    

    
 
    st.subheader("Logical Regression Model")
    
    st.subheader("Confusion Matrix (Logical Regression)")
    st.image("../Pics/confusion-matrix-logical-reg.png", caption="", width=800)
    
    st.subheader("")
    
    
    st.subheader("Classification Report (Logical Regression)")
    st.image("../Pics/classification-report-log-reg.png", caption="", width=700)
    
    st.subheader("")
    
    st.subheader("Analysis of logical regression model")
    st.write("1. heavily biased towards home wins")
    st.write("2. Really bad at predicting draws")
    st.write("3. accuracy is misleading")
    
    
    
    
    
    st.subheader("Random Forest Model")
    st.subheader("Confusion Matrix (Random Forest)")
    st.image("../Pics/confusion-matrix-random-forest.png", caption="", width=800)
    
    st.subheader("")
    
    st.subheader("Classification Report (Random Forest)")
    st.image("../Pics/classification-report-random-forest.png", caption="", width=700)
    
    st.subheader("")
    
    st.subheader("Analysis of random forest model")
    st.write("1. still biased towards home wins but less than logical regression")
    st.write("2. better at predicting draws")
    st.write("3. accuracy is misleading")
    
    st.subheader("")

    
    #st.image("../Pics/feature-importances-random-forest.png", caption="", width=900)
    #st.subheader("")
    
    
    st.subheader("So can we predict the outcome of a match?")
    st.write("1. ")
    st.write("1. ")
    
    
    
    
    
    
    
    