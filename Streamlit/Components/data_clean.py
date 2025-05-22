import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import LabelEncoder

def data_cleaning():
    st.title('Data cleaning')
    st.write('We are working with one dataset that shows different data about football bettings in the Premier League.')
    
    # Load data
    fb_df = pd.read_csv('../data/Premier-League-2015-2019.csv')
    
    st.subheader("Dataset Info")
    buffer = io.StringIO()
    fb_df.info(buf=buffer)
    st.text(buffer.getvalue())
    
    st.subheader("Missing & Duplicates Overview")
    st.write("Duplicated rows:")
    st.write(fb_df.duplicated().sum())
    
    st.write("Missing values per column:")
    st.dataframe(fb_df.isnull().sum().reset_index().rename(columns={0: "Missing Values", "index": "Column"}))
    
    # Rename columns
    fb_df.rename(columns={
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'outcome_by_HOME': 'home_outcome',
        'AVERAGE_ODD_WIN': 'avg_odd_home_win',
        'AVERAGE_ODD_DRAW': 'avg_odd_draw',
        'AVERAGE_ODD_OPPONENT_WIN': 'avg_odd_away_win',
        'RANKINGHOME': 'home_ranking',
        'RANKING AWAY': 'away_ranking',
        'LAST_GAME_RHOMETEAM': 'last_home_result',
        'LAST_GAME_RAWAYTEAM': 'last_away_result'
    }, inplace=True)
    
    st.subheader("Data Preview")
    st.dataframe(fb_df.head())
    
    st.title("Label Encoding")
    st.write("Label encode home and away teams so they match.")
    
    all_teams = pd.concat([fb_df['home_team'], fb_df['away_team']]).unique()
    team_encoder = LabelEncoder()
    team_encoder.fit(all_teams)

    fb_df['home_team'] = team_encoder.transform(fb_df['home_team'])
    fb_df['away_team'] = team_encoder.transform(fb_df['away_team'])
    
    st.title("Label Encoding for home_outcome")
    fb_df['home_outcome'] = fb_df['home_outcome'].map({'L': 0, 'D': 1, 'W': 2})
    st.dataframe(fb_df[['home_team', 'away_team', 'home_outcome']].sample(10))
    
    st.title("One-Hot Encoding")
    st.write("Example of how one-hot encoding could be done:")
    st.code("df_onehot = pd.get_dummies(fb_df, columns=['home_outcome'])")
    
    st.write("You may need to rerun to see changes due to caching in Streamlit.")

    st.title("Fixing Bloated Rankings")
    st.write("Rankings with value 20 may include values for seasons 18–19. These are redistributed randomly.")

    def redistribute_twenty(column):
        mask = fb_df[column] == 20
        n = mask.sum()
        replacements = np.array([18, 19, 20] * (n // 3 + 1))[:n]
        np.random.shuffle(replacements)
        fb_df.loc[mask, column] = replacements

    redistribute_twenty('home_ranking')
    redistribute_twenty('away_ranking')

    st.write("Home Ranking Value Counts:")
    st.dataframe(fb_df['home_ranking'].value_counts().sort_index())

    st.write("Away Ranking Value Counts:")
    st.dataframe(fb_df['away_ranking'].value_counts().sort_index())

    st.title("Outlier Detection")
    st.write("Summary statistics before removing outliers:")
    st.dataframe(fb_df[['avg_odd_home_win', 'avg_odd_draw', 'avg_odd_away_win']].describe())

    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    for col in ['avg_odd_home_win', 'avg_odd_draw', 'avg_odd_away_win']:
        fb_df = remove_outliers_iqr(fb_df, col)

    st.write("After removing outliers:")
    st.dataframe(fb_df[['avg_odd_home_win', 'avg_odd_draw', 'avg_odd_away_win']].describe())

    st.success("Data cleaned and ready for modeling or visualization.")

    # Optionally save cleaned data
    fb_df.to_csv('../data/cleaned-premier-league-data.csv', index=False)
