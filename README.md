[Link to Kaggle Dataset1](https://www.kaggle.com/datasets/ivanpv/premier-league-football-matches-20152019/data?select=Premier-League-2015-2019_TRAINING.csv)

[Link to Kaggle Dataset2](https://www.kaggle.com/datasets/idoyo92/epl-stats-20192020?resource=download)
[Link to Kaggle Dataset3](https://www.kaggle.com/datasets/sanjeetsinghnaik/premier-league-matches-20142020/data)
---

## Problem Statement

Football matches can be hard to predict, but maybe there are some patterns that can help.  
We want to find out if things like home advantage, past rankings, or betting odds can tell us something useful about which team is more likely to win.

---

## 1. Which machine learning methods did you choose to apply in the application and why?

We made use of: **Linear Regression**, **Decision Tree**, and **K-Means** models.

- **Linear Regression** – _Christian_
- **Decision Tree** – _Tobias_
- **K-Means**: Used to discover patterns in the betting odds and group similar matches.

---

## 2. How accurate is your solution of prediction? Explain the meaning of the quality measures.

- **Linear Regression** – _Christian_
- **Decision Tree** – _Tobias_
- **K-Means**:  
  Silhouette Score: **0.470**, which indicates a **moderate clustering quality**.

---

## 3. Which features have the most impact on the outcome of a match?

- Odds had the highest impact on matches, followed by last season rankings.  
- A team's last match result had surprisingly little impact on their next match.

---

## 4. What could be done for further improvement of the accuracy of the models?

- **Linear Regression** – _Christian_
- **Decision Tree** – _Tobias_
- **K-Means**:  
  Including more features may help the model identify better patterns.  
  We could also experiment with different numbers of clusters to find better groupings.

---

## 5. Something found during data exploration – _Umair_
_(Add your findings here)_

---

## 6. Do home teams have an advantage over the away team? – _Christian_

_(Add answer here)_

---

## 7. add something you found tobias – _tobias_

_(Add answer here)_

---

## 8. Can the odds be put into groups?

Yes, with the help of clustering, the odds can be grouped into four clusters:

1. Where the home team is favored  
2. Where the away team is favored  
3. Balanced matches  
4. Outliers that don’t fit into the other groups

---

## 9. Which were the challenges in the project development?

Yes, we encountered several challenges:

1. **Missing rankings**: Team rankings 18 and 19 were completely missing, while rank 20 appeared approximately three times more often than other ranks. This suggests that the missing ranks may have been incorrectly grouped under rank 20.

2. **Unexpected betting behavior**: While match outcomes suggest that home teams are more likely to win (indicating a home advantage), the betting odds do not reflect this. This may be due to errors in the dataset or the result of public betting behavior, introducing noise into the data.
