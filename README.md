link to keggle: https://www.kaggle.com/datasets/ivanpv/premier-league-football-matches-20152019/data?select=Premier-League-2015-2019_TRAINING.csv

PROBLEM STATEMENT:
Football matches can be hard to predict, but maybe there are some patterns that can help. We want to find out if things like home advantage, past rankings, or betting odds can tell us something useful about which team is more likely to win.


1. Which machine learning methods did you choose to apply in the application and why?
we made use of: linear regression, decision tree and k-means models

linear regression: christian

decision tree: tobias

K-Means was used to discover patterns in the betting odds and group similar matches


2. How accurate is your solution of prediction? Explain the meaning of the quality measures.

linear regression: christian

decision tree: tobias

k-means: Silhouette Score: 0.470, which indicates a moderate clustering quality.

3. Which features have the most impact on the outcome of a match?
odds had the highest impact on matches followed by last season rankings. 
Team’s last match result had surprisingly little impact on their next match. 

4. What could be done for further improvement of the accuracy of the models?

linear regression: christian

decision tree: tobias

k-means: including more features may help the model see better patterns.
We could also experiment with different numbers of clusters to find better groupings.

5. noget du fandt i din exploration (umair)


6. do home teams have an advantage over the away team (christian)


7. Are employees of different gender paid equally in all departments? (tobias)

8. Can the odds be put into groups?

yes with help of clustering can they be put into 4 clusters
1. where the home team is favored
2. where the away team is favored
3. balanced teams
4. outliers that doesn’t fit the other groups

9. Which were the challenges in the project development?
Yes, we encountered several challenges:
while the data didn’t require much cleaning there where a few issues with it, such as 
1. Team rankings 18 and 19 were completely missing, while rank 20 appeared approximately three times more often than other ranks. This suggests that the missing ranks may have been incorrectly grouped under rank 20.
2. While match outcomes suggest that home teams are more likely to win (indicating a home advantage), the betting odds do not reflect this. it may be due to errors in the dataset or could be the result of betting behavior, introducing noise into the data.

