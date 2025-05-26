import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import io

def load_eda():
    st.title("Exploratory Data Analysis")
    st.write("The data is loaded from the cleaned CSV file `cleaned-premier-label.csv` and displayed to get an overview of its contents.")
    
    df = pd.read_csv('../data/cleaned-premier-label.csv')
    st.dataframe(df)
    st.write("Checking the shape to view the number of rows and columns in the dataset.")
    st.dataframe(df.shape)
    st.write("Dataset Information Overview")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write("Checking the descriptive statistics of the dataset's numerical columns, such as mean, standard deviation, min, max, and quartiles.")
    st.dataframe(df.describe())
    st.write("Calculating the Mean of All Numerical Columns")
    column_means = df.mean(numeric_only=True)
    st.dataframe(column_means)
    st.write("Calculating the Median of All Numerical Columns")
    column_medians = df.median(numeric_only=True)
    st.dataframe(column_medians)
    st.write("Calculating the Mode of Each Column")
    single_mode = df.mode().iloc[0]
    st.dataframe(single_mode)
    st.subheader("Histogram and KDE Plot for Betting Odds")
    st.write("For each of the numerical columns related to betting odds (`avg_odd_home_win`, `avg_odd_draw`, and `avg_odd_away_win`), a histogram with a Kernel Density Estimate (KDE) is plotted. These visualizations help in understanding the distribution and spread of betting odds in the dataset.")
    numerical_cols = ['avg_odd_home_win', 'avg_odd_draw', 'avg_odd_away_win']

    for column in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[column], bins=30, kde=True)
        plt.title(f'Histogram + KDE for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
    
        st.pyplot(plt.gcf()) 
        plt.clf()
    
    st.subheader(" Interpretation of the Histograms with KDE for Betting Odds")
    st.write("""
Each plot displays the distribution of a specific betting odd type:

- **Home Win Odds (`avg_odd_home_win`)**:  
  The histogram shows the frequency of average odds for home team victories. Most values are concentrated between roughly 1.5 and 2.5, indicating that home teams are often favored to win, but not overwhelmingly.

- **Draw Odds (`avg_odd_draw`)**:  
  This distribution is generally more centered, often peaking around 3.0. This reflects that draws are considered moderately likely but less frequent than home wins, according to the betting markets.

- **Away Win Odds (`avg_odd_away_win`)**:  
  These odds tend to be higher, often peaking above 3.0 and stretching further. This indicates that away teams are usually considered underdogs, and higher odds are offered for away victories.

The KDE (Kernel Density Estimate) line over each histogram smooths out the distribution, making it easier to observe the central tendency and skewness of each odds type:
- If the KDE is skewed to the right (long tail to the right), higher odds occur less frequently.
- If it's more symmetric, the outcomes are more evenly distributed.
""")
    
    st.subheader("Correlation Heatmap of Numerical Features")
    st.write("Shows the correlation between numerical variables in the dataset.")
    
    st.subheader("Correlation Heatmap")

    # Load the dataset
    df_new = pd.read_csv('../data/cleaned-premier-label.csv')

    # Keep only numerical columns
    only_numerical_df = df_new.select_dtypes(include='number')

    # Calculate correlation matrix
    corr = only_numerical_df.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Korrelation mellem kampdata og resultater i Premier League')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(plt)
    
    st.write("""
             
             ### Key Insights from the Correlation Heatmap

#### 1. **Odds Reflect Match Outcomes**
- **`home_outcome` is negatively correlated with `avg_odd_home_win` (-0.26)** and positively with `avg_odd_away_win` (0.29).
  - This means that when the odds for a home win are low (home team is favored), the home team tends to win more often — which aligns with typical betting logic.

#### 2. **Home and Away Odds Are Strongly Linked**
- **`avg_odd_home_win` and `avg_odd_away_win` show a strong negative correlation (-0.48)**:
  - If the home team is strongly favored (low odds), the away team is considered less likely to win (high odds), and vice versa.

#### 3. **Cumulative Match Statistics Are Interrelated**
- There are strong positive correlations between:
  - `home_total_wins_so_far`, `home_total_draws_so_far`, and `home_total_losses_so_far` (0.53–0.71),
  - The same applies to the away team's stats.
  - This makes sense, as all results accumulate over the season — more matches played means more of each result type.

#### 4. **Stronger Teams Reduce Opponent Odds**
- **`home_ranking` is negatively correlated with `avg_odd_away_win` (-0.48)**:
  - A lower ranking number (i.e., a higher-ranked team) correlates with lower chances for the away team to win, hence higher away win odds.

             
             """)
    
    st.write("""
             
             ## Applying PCA on Numerical Data

PCA (Principal Component Analysis) is performed on all numerical columns to reduce dimensionality and explore variance structure.

             
             """)
    
    # Keep only numeric columns for PCA
    numeric_df = df.select_dtypes(include='number')

    # Fit PCA on numeric data
    pca = PCA()
    pca_data = pca.fit_transform(numeric_df)

    st.dataframe(pca_data)
    
    st.write("""
             
             ## Explained Variance Ratio of Principal Components

Displays the proportion of total variance explained by each principal component.  
This helps determine how many components are needed to capture most of the information in the dataset.

             
             """)
    
    # The PCA class contains explained variance  ratio, 
    # which returns the variance caused by each of the principal components
    explained_variance = pca.explained_variance_ratio_  
    st.dataframe(explained_variance)
    
    st.write("""
             
             ## Scree Plot of Principal Component Variance

A scree plot showing how much variance each principal component explains.  
This visualization helps identify the "elbow point" — the optimal number of components to retain for dimensionality reduction.

             
             """)
    
    # Create a fresh figure and axes
    fig, ax = plt.subplots()

    # Plot the explained variance
    ax.plot(explained_variance, 'bx-')
    ax.set_xlabel('Component')
    ax.set_ylabel('Variance')
    ax.set_title('The optimal number of components')

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    st.write("""
             
             ### Interpretation of the Scree Plot

The scree plot visualizes the explained variance of each principal component. Key observations:

- **The first component explains ~39%** of the total variance — the most informative.
- **The second and third components explain ~22% and ~19%**, respectively.
- After the third component, the explained variance drops significantly, indicating diminishing returns.

The "elbow point" occurs around **component 3**, suggesting that **the first 3 components capture most of the meaningful variance** in the data. 
             
             """)
    
    
    st.write("""
             
             ## Cumulative Explained Variance

This plot shows how the total explained variance accumulates as more principal components are added.  
It helps determine the number of components needed to retain a desired amount of total variance (e.g., 90%).

             
             """)
    
    # Calculate cumulative explained variance
    cumulative = np.cumsum(explained_variance)

    # Create a fresh figure and axes
    fig, ax = plt.subplots()

    # Plot the cumulative variance
    ax.plot(cumulative, 'b*-')
    ax.set_xlabel('Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Cumulative Explained Variance by Component')

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    
    st.write("""
             
             ### Interpretation of the Cumulative Explained Variance Plot

This plot shows how much of the total variance is captured as more principal components are added.

#### Key Observations:
- The **first 3 components explain over 80%** of the total variance.
- With **5 components**, the cumulative explained variance exceeds **90%**.
- After that, the curve flattens, meaning additional components add minimal new information.

#### Conclusion:
Selecting the **first 3 to 5 components** is likely sufficient to retain most of the information in the data while reducing dimensionality significantly.

             
             """)
    
    st.write("""
             
             ## Boxplots of Betting Odds

Displays individual boxplots for home win, draw, and away win odds.  
Boxplots help identify the distribution, central values, and potential outliers for each odds type.

             
             """)
    
    # Select the columns to plot
    columns_to_plot = ['avg_odd_home_win', 'avg_odd_draw', 'avg_odd_away_win']

    # Create a figure and axes for subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))  # Adjust layout as needed

    # Plot each boxplot separately
    for i, column in enumerate(columns_to_plot):
        df[column].plot.box(ax=axes[i], whis=1.5)
        axes[i].set_title(f'Boxplot of {column}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Odds')

    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)
    
    
    st.write("""
             
             ### Interpretation of Boxplots for Betting Odds

The boxplots show the distribution of average betting odds for home wins, draws, and away wins:

- **`avg_odd_home_win`**:
  - The median is around 6.8.
  - There are several **outliers above 10**, indicating rare matches where the home team was a strong underdog.
  - The interquartile range is relatively wide, showing moderate variability.

- **`avg_odd_draw`**:
  - A more **symmetric and centered** distribution with a median around 3.5.
  - Less variation and fewer outliers compared to the other odds types.

- **`avg_odd_away_win`**:
  - Lower median (~3.0), but a high number of **outliers above 7**, suggesting that away teams are often considered underdogs.
  - The distribution is **right-skewed**, indicating more high-value odds.

#### Summary:
- **Draw odds** are the most stable and predictable.
- **Away win odds** have the most extreme values (outliers).
- These **outliers** may be worth further investigation, as they could indicate surprising or unusual match outcomes.

             
             """)
    
    st.write("""
             
             ## Research Questions 1
             ### How do bookmaker odds correlate with actual match outcomes?
             
             """)
    
    df = pd.read_csv('../data/cleaned-premier-onehot.csv')
    # Calculate correlation of odds with their respective outcomes
    home_win_corr = df['avg_odd_home_win'].corr(df['home_outcome_W'])
    draw_corr = df['avg_odd_draw'].corr(df['home_outcome_D'])
    away_win_corr = df['avg_odd_away_win'].corr(df['home_outcome_L'])

    st.write("Correlation between specific odds and corresponding outcomes:")
    st.write(f"Home win odds vs actual home wins:   {home_win_corr:.3f}")
    st.write(f"Draw odds vs actual draws:           {draw_corr:.3f}")
    st.write(f"Away win odds vs actual away wins:   {away_win_corr:.3f}")
    
    
    # Define labels and calculate correlations
    labels = ['Home Win Odds', 'Draw Odds', 'Away Win Odds']
    correlations = [
        df['avg_odd_home_win'].corr(df['home_outcome_W']),
        df['avg_odd_draw'].corr(df['home_outcome_D']),
        df['avg_odd_away_win'].corr(df['home_outcome_L'])
]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the bar chart
    bars = ax.bar(labels, correlations, color=['#4CAF50', '#FFC107', '#F44336'])

    # Annotate bars with correlation values
    for bar in bars:
         height = bar.get_height()
         ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}", ha='center', va='bottom')

    # Set titles and formatting
    ax.set_title("Correlation Between Odds and Corresponding Match Outcomes")
    ax.set_ylabel("Correlation Coefficient")
    ax.set_ylim(min(correlations) - 0.05, max(correlations) + 0.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Tight layout and render in Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("""
             
             ### Home Win Odds

A correlation of -0.208 means that as home win odds decrease (i.e., the home team is more favored), the likelihood of a home win increases.

This is expected and shows that bookmaker odds have some predictive value for home wins.

---

### Away Win Odds

A correlation of -0.253 is slightly stronger, but otherwise shows the same pattern as home win odds: lower odds → higher likelihood of that outcome.

Although it's slightly surprising that away win odds, on average, perform better, since home teams are generally favored, we might have expected home odds to be more predictive.

---

### Draw Odds

A correlation of -0.095 is very weak, suggesting that draw odds are poor predictors of actual draws.

---

### Conclusion

Bookmaker odds do reflect actual match outcomes to some extent, especially for wins and losses.

However, the predictive strength is moderate at best, and draw prediction remains weak.

             
             """)
    
    st.write("""
             
             ## Research Questions 2
             ### Does a higher ranking from the previous season increase the likelihood of winning?
             
             """)
    
    # Correlation: Does a better home ranking increase home win chances?
    home_corr = df['home_ranking'].corr(df['home_outcome_W'])
    away_corr = df['away_ranking'].corr(df['home_outcome_W'])

    st.write("Correlation between last season's rankings and home wins:")
    st.write(f"Home ranking vs home win:  {home_corr:.2f}")
    st.write(f"Away ranking vs home win:  {-away_corr:.2f}  # flipped to represent 'weaker away team'")

    # Average ranking per outcome
    st.write("\nAverage team rankings per outcome:")
    st.write("When home team wins:")
    st.write("Home ranking:", round(df[df['home_outcome_W'] == 1]['home_ranking'].mean(), 2))
    st.write("Away ranking:", round(df[df['home_outcome_W'] == 1]['away_ranking'].mean(), 2))

    st.write("\nWhen match is a draw:")
    st.write("Home ranking:", round(df[df['home_outcome_D'] == 1]['home_ranking'].mean(), 2))
    st.write("Away ranking:", round(df[df['home_outcome_D'] == 1]['away_ranking'].mean(), 2))

    st.write("\nWhen away team wins:")
    st.write("Home ranking:", round(df[df['home_outcome_L'] == 1]['home_ranking'].mean(), 2))
    st.write("Away ranking:", round(df[df['home_outcome_L'] == 1]['away_ranking'].mean(), 2))
    
    st.write("""
             
             ### Analysis of Team Rankings and Match Outcomes

These values suggest a small but consistent trend:

Better-ranked teams (from last season) are more likely to win, but ranking alone is not a strong predictor.

In matches where the home team won, they were on average better ranked than their opponent.

In away wins, the away team had a significantly better ranking than the home team.

Draws tend to occur when teams are closely matched in ranking.

### Conclusion

While previous season rankings do influence match outcomes, the effect is relatively weak on its own.

             
             """)
    
    df['ranking_diff'] = df['away_ranking'] - df['home_ranking']

    # Bin the ranking difference into intervals
    df['ranking_diff_bin'] = pd.cut(df['ranking_diff'], bins=[-20, -10, -5, 0, 5, 10, 20])

    bin_stats = df.groupby('ranking_diff_bin', observed=False)[['home_outcome_L', 'home_outcome_D', 'home_outcome_W']].mean().reset_index()

    # Rename for clarity
    bin_stats.columns = ['ranking_diff_bin', 'away_win_rate', 'draw_rate', 'home_win_rate']
    
    
    # Create a Streamlit-friendly plot for outcome rates by ranking difference
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the outcome rates
    ax.plot(bin_stats['ranking_diff_bin'].astype(str), bin_stats['home_win_rate'], label='Home Win Rate', marker='o')
    ax.plot(bin_stats['ranking_diff_bin'].astype(str), bin_stats['draw_rate'], label='Draw Rate', marker='o')
    ax.plot(bin_stats['ranking_diff_bin'].astype(str), bin_stats['away_win_rate'], label='Away Win Rate', marker='o')

    # Set plot titles and labels
    ax.set_title("Match Outcome Rates vs. Ranking Difference (Away - Home)")
    ax.set_xlabel("Ranking Difference (Away - Home)")
    ax.set_ylabel("Proportion of Outcomes")
    ax.legend()
    ax.grid(True)

    # Use tight layout and show it in Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("""
             
             ## Research Questions 3
             ### Does the home team have an advantage over the away team?
             
             """)
    
    total_matches = len(df)

    home_wins = df['home_outcome_W'].sum()
    draws = df['home_outcome_D'].sum()
    away_wins = df['home_outcome_L'].sum()

    home_win_rate = home_wins / total_matches
    draw_rate = draws / total_matches
    away_win_rate = away_wins / total_matches

    st.write("Match Outcome Rates:")
    st.write(f"Home win rate:  {home_win_rate:.2%}")
    st.write(f"Draw rate:      {draw_rate:.2%}")
    st.write(f"Away win rate:  {away_win_rate:.2%}")

    if away_win_rate > 0:
        st.write(f"\nHome win to away win ratio: {home_win_rate / away_win_rate:.2f}")
        
        
    home_wins = df['home_outcome_W'].sum()
    draws = df['home_outcome_D'].sum()
    away_wins = df['home_outcome_L'].sum()

    labels = ['Home Wins', 'Draws', 'Away Wins']
    values = [home_wins, draws, away_wins]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=['#4CAF50', '#FFC107', '#F44336'])

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.0f}", ha='center', va='bottom')

    ax.set_title("Match Outcome Distribution: Home Advantage Analysis")
    ax.set_ylabel("Number of Matches")
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("""
             
             ### conclusion
Our data shows the home team has a clear home advantage over the away team:
             
             """)
    
    st.write("""
             
             # Reasearch question 4: 
             ## Do teams that won their last game have a higher probability of winning their current game?
             
             """)
    
    home_result_analysis = df.groupby("last_home_result")["home_outcome_W"].mean().reset_index()
    home_result_analysis.columns = ["Last Home Result", "Home Win Rate"]

    # Apply readable labels
    home_result_analysis["Last Home Result"] = home_result_analysis["Last Home Result"].map({
       -1:"Lost Last Game",
        0:"Drew Last Game",
        1:"Won Last Game"
})

    st.write(home_result_analysis)
    
    away_result_analysis = df.groupby("last_away_result")["home_outcome_L"].mean().reset_index()
    away_result_analysis.columns = ["Last Away Result", "Away Win Rate"]

    # Apply readable labels
    away_result_analysis["Last Away Result"] = away_result_analysis["Last Away Result"].map({
     -1: "Lost Last Game",
      0: "Drew Last Game",
      1: "Won Last Game"
})

    st.write(away_result_analysis)
    
    
    st.write("""
             
             
             We grouped matches by the outcome of each team's previous game (win, draw/first game, or loss) and calculated the win rate for the current match:

#### Home Teams

| Last Home Result | Home Win Rate |
|------------------|----------------|
| Lost Last Game   | 41.0%          |
| Drew Last Game   | 45.8%          |
| Won Last Game    | 45.5%          |

#### Away Teams

| Last Away Result | Away Win Rate |
|------------------|----------------|
| Lost Last Game   | 27.2%          |
| Drew Last Game   | 30.2%          |
| Won Last Game    | 33.2%          |

---

### Interpretation

- Home teams that lost their previous match had the lowest win rate (41.0%).
- Home teams that drew or won their last match had similar win rates, around 45.5%–45.8%.

- Away teams that lost last time had only a 27.2% win rate.
- Away teams that won last time had the highest win rate at 33.2%.

---

### Conclusion

For home teams, the result of the previous match does not seem to have a strong influence on the current game's outcome. Winning or drawing gives a slightly higher chance than losing, but the difference is marginal.

For away teams, there is a more noticeable pattern. Teams that won their last game are more likely to win again, while those who lost are less likely to win the next match. This suggests that recent performance may have a stronger influence on away teams than home teams.

             
             
            --------------------------------------------------------------------------- """)
    
    st.write("""
             
             # Reasearch question 5:
             ## Can we accurately predict the outcome of a match using our data, and which machine learning model performs best for this task?
             
            ---------------------------------------------------------------------------------------------------------- """)
    
    
    
    
    def label_outcome(row):
        if row['home_outcome_W'] == 1:
            return 'Home Win'
        elif row['home_outcome_D'] == 1:
            return 'Draw'
        else:
            return 'Away Win'

    df['actual_result'] = df.apply(label_outcome, axis=1)

    odds_by_outcome = df.groupby('actual_result')[['avg_odd_home_win', 'avg_odd_draw', 'avg_odd_away_win']].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    odds_by_outcome.plot(kind='bar', ax=ax)

    ax.set_title("Average Bookmaker Odds by Actual Match Outcome")
    ax.set_ylabel("Average Odds")
    ax.set_xlabel("Actual Match Outcome")
    ax.legend(["Home Win Odds", "Draw Odds", "Away Win Odds"])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("Shows the average bookmaker odds for each possible outcome, in matches that ended in a home win, draw, or away win.")
    
    st.write("""
             
             ### Odds vs Reality

This again highlights a strange pattern in our dataset:  
Even though home teams, on average, are more likely to win, this is not reflected in the odds data.

The fact that bookmakers almost never favor the home team suggests there may be an underlying issue or bias in the dataset.

It could also reflect how gamblers tend to bet, if the betting market disproportionately favors away teams or draws, bookmakers might adjust odds accordingly to balance their risk.  
This market-driven skew could explain why the odds don't align with the actual win rates of home teams.

Either way, this discrepancy may indicate noise or bias in the data that could affect model performance.

             
           """)
    
    
    
    def favored_outcome(row):
        odds = {
            'Home Win': row['avg_odd_home_win'],
            'Draw': row['avg_odd_draw'],
            'Away Win': row['avg_odd_away_win']
    }
        return min(odds, key=odds.get)

    df['bookmaker_favored'] = df.apply(favored_outcome, axis=1)

    favored_counts = df['bookmaker_favored'].value_counts(normalize=True) * 100
    st.write("How often each outcome was favored by the odds (lowest odds):")
    st.write(favored_counts.round(2))
    
    actual_outcomes = df[['home_outcome_W', 'home_outcome_D', 'home_outcome_L']].sum()
    actual_outcomes.index = ['Home Win', 'Draw', 'Away Win']
    actual_outcomes_percent = actual_outcomes / actual_outcomes.sum() * 100

    st.write("\nActual outcome distribution:")
    st.write(actual_outcomes_percent.round(2))
    
    df['ranking_diff'] = df['away_ranking'] - df['home_ranking']

    avg_diff_when_away_favored = df[df['bookmaker_favored'] == 'Away Win']['ranking_diff'].mean()
    st.write(f"\nAverage ranking_diff when Away Win is favored: {avg_diff_when_away_favored:.2f}")
    
    st.write("""
             
             Bookmakers systematically favor away teams
Despite home teams winning more often, bookmakers set away win odds lower in 59% of all matches, while home teams are only favored in 2%.

even though home teams wins 13.32% more often
             
             """)
    

    







    
    

    
               

