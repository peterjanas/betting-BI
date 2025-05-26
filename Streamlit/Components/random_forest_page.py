# Components/naive_bayes_page.py
import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("../Models/myrandomforest.pkl")

def show_rf_model_page():
    st.title("Random Forest Classifier")

    model = load_model()

    st.write("Enter the relevant features to predict match outcome:")

    feature1 = st.number_input("avg_odd_home_win")
    feature2 = st.number_input("avg_odd_draw")
    feature3 = st.number_input("avg_odd_away_win")
    feature4 = st.number_input(" home_ranking")
    feature5 = st.number_input("away_ranking")
    feature6 = st.number_input("home_seasons_wins_so_far")
    feature7 = st.number_input("home_seasons_draws_so_far")
    feature8 = st.number_input("home_seasons_losses_so_far")

    if st.button("Predict"):
        input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]])
        prediction = model.predict(input_data)

    # Map numeric prediction to label
        outcome_map = {
        0: "Loss",
        1: "Draw",
        2: "Win"
     }

        predicted_label = outcome_map.get(prediction[0], "Unknown")
        st.success(f"Prediction: {predicted_label}")

