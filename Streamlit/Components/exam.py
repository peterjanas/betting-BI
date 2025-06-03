import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def load_exam():


    st.title("BI project: premier league and betting odds")

  
    df = pd.read_csv('../data/cleaned-premier-label.csv')
    
    st.subheader("What we wanted to research")
    st.write("This dataset contains information about Premier League matches from the 2015-2019 seasons, including team statistics, match outcomes, and betting odds.")
    
    col1, col2 = st.columns(2)

    with col1:
        st.image("../Pics/histogram-home-win.png", caption="Home Win Histogram", width=700)

    with col2:
        st.image("../Pics/histogram-away-win.png", caption="Away Win Histogram", width=700)
        
    st.write("note slet extra tekst på histogram")
    
    
    st.image("../Pics/match-outcome-distribution.png", caption="", width=1000)
    st.write("note skal også havde % af dem")
    
    
    
    
    st.image("../Pics/averge-odds-by-outcome.png", caption="", width=1200)
    
    
    st.write("note: skal havde den med odds favored % med")
    
    
    
    st.image("../Pics/correlation-between-odds-and-outcome.png", caption="", width=1000)
    
    
    
    st.image("../Pics/outcome-versus-ranking-difference.png", caption="", width=1200)
    
    
    
    
    st.subheader("models")
    
    
    st.write("skal havde classification report med")
    
    
    st.image("../Pics/confusion-matrix-logical-reg.png", caption="", width=900)
    
    st.write("note: analyse")
    
    
    
    
    
    st.image("../Pics/confusion-matrix-random-forest.png", caption="", width=900)
    
    st.image("../Pics/feature-importances-random-forest.png", caption="", width=900)
    
    
    
    
    