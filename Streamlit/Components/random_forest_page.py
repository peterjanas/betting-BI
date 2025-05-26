# Components/naive_bayes_page.py

import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("../Models/myrandomforest.pkl")


def show_randomforest_model_page():
    st.title("randomforestmodel")
    model = load_model()

    st.write("enter relevant data to to predict match outcome")

    ranking_diff = st.number_input("ranking_diff")
    avg_odd_home_win = st.number_input("avg_odd_home_win")
    avg_odd_draw = st.number_input("avg_odd_draw")
    avg_odd_away_win = st.number_input("avg_odd_away_win")
    home_wins = st.number_input("home_season_wins_so_far")
    home_draws = st.number_input("home_season_draws_so_far")
    home_losses = st.number_input("home_season_losses_so_far")
    away_wins = st.number_input("away_season_wins_so_far")
    away_draws = st.number_input("away_season_draws_so_far")

    if st.button("Predict"):
        input_data = np.array([[ranking_diff, avg_odd_home_win, avg_odd_draw, avg_odd_away_win, home_wins, home_draws, home_losses, away_wins,away_draws]])
        prediction = model.predict(input_data)


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

        predicted_label = outcome_map.get(prediction[0],"unknown")
        st.success(f"Prediction: {predicted_label}")
        if hasattr(model,"predict_proba"):
            probs = model.predict_proba(input_data)
            st.write("Class probabilities:")
            for i, prob in enumerate(probs[0]):
                st.write(f"{outcome_map.get(i)}: {prob:.2f}")                


        predicted_label = outcome_map.get(prediction[0], "Unknown")
        st.success(f"Prediction: {predicted_label}")


