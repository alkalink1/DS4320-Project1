# DS 4320 Project 1: Predicting NCAA Women’s Basketball Tournament Outcomes

**Executive Summary:**

This project predicts outcomes of women’s NCAA tournament basketball games using regular season performance metrics and tournament seeding. By combining historical game data with engineered features such as scoring differentials and win percentage, the model identifies key factors that influence tournament success. The goal is to demonstrate how data-driven insights can support competitive strategy and decision-making in sports.

Alka Link - eju2pk

DOI

[Press Release] Link

[Data] Link

[Pipeline] Link

[MIT License](https://github.com/alkalink1/databydesignproject1/blob/main/LICENSE)

## Problem Definition
**General Problem:**

Predicting sports games outcomes

**Specific Problem Statement:**

The refined problem is to predict the outcome of NCAA college basketball games using historical game results and team performance statistics. By analyzing patterns in past games, like scoring margins, win-loss records, and game location, we aim to estimate the probability that a team will win a given matchup. The project will use historical NCAA basketball data, including regular season and tournament results, to explore how statistical patterns can be used to predict future game outcomes.


**Rationale for Project Refinement:**

The general problem of predicting sports game outcomes is extremely broad because many factors can influence the result of a game, including player injuries, team strategies, coaching decisions, and random variation. To make the problem more manageable, the project focuses specifically on NCAA college basketball games and uses structured historical data to model outcomes. This refinement allows the analysis to focus on measurable variables such as scoring margins, game location, and historical performance trends. By narrowing the problem to college basketball outcomes and using historical game statistics, the project becomes a clearly defined data analysis problem that can be explored using statistical and machine learning methods.


**Motivation:**

Predicting sports game outcomes is an important application of sports analytics. Teams, analysts, and fans increasingly rely on statistical models to evaluate team performance and estimate the probability of winning future games. In college basketball, predictive models are widely used during the NCAA tournament, where analysts attempt to forecast game outcomes for brackets and tournament simulations. By analyzing historical game data, analysts can identify patterns that help explain why some teams perform better than others. Developing predictive models for game outcomes can provide insights into the factors that most strongly influence winning and demonstrate how data can be used to support decision-making in sports.


Headline of Press Release: Link

## Domain Exposition
Terminology:

Domain:

Background Reading: 

Table of Readings: 

## Data Creation

Provenance: 

For this project, I used the Kaggle Women’s NCAA Basketball dataset to build a predictive dataset for tournament game outcomes. My project goal is to predict whether a team wins an NCAA tournament game using historical performance data, so I selected files that contain regular season results, tournament results, and tournament seed information. Specifically, I used WRegularSeasonDetailedResults.csv for regular season team performance, WNCAATourneyCompactResults.csv for tournament matchups and outcomes, and WNCAATourneySeeds.csv for tournament seed information. This dataset choice matches the project objective of predicting NCAA basketball game outcomes using historical team statistics.

To create the final modeling dataset, I first computed team-level regular season statistics by season. I calculated each team’s average score using both games they won and games they lost, then combined those into a single average scoring feature. I also calculated each team’s win percentage from regular season wins and losses. Next, I extracted each team’s tournament seed and converted the seed code into a numeric seed value. After that, I created a matchup-level tournament dataset by taking each NCAA tournament game and representing it from both team perspectives: one row where Team1 is the winner and one row where Team1 is the loser. This created a balanced binary outcome variable called Win. Finally, I merged team-level features into each matchup for Team1 and Team2 and engineered difference-based features (ScoreDiff, WinPctDiff, and SeedDiff) so the model could learn relative team strength rather than raw team values. The final dataset is therefore game-level, prediction-oriented, and constructed entirely from pre-game information, avoiding data leakage.


Code to create data: 

| File Name                         | Description                                                                                                                                                                              | Link                |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| data_pipeline_and_model.ipynb     | Loads Kaggle NCAA datasets, creates team-level regular season features, merges them into tournament matchups, engineers difference variables, and trains/evaluates classification models | [Colab notebook link](https://colab.research.google.com/drive/1wo1f_aWnYOD9oRVbx0zrrs_nh9Ayw6Yd?usp=sharing) |
| WRegularSeasonDetailedResults.csv | Raw regular season game-level results used to compute average score and win percentage                                                                                                   | [Local/Kaggle file](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data?inquiry-id=inq_nP8ZBe1wnwQZXjGDRx2xgjp4iY7Y&reference-id=25013496&subject=25013496&status=approved&fields%5Bname-first%5D%5Btype%5D=string&fields%5Bname-first%5D%5Bvalue%5D=&fields%5Bname-middle%5D%5Btype%5D=string&fields%5Bname-middle%5D%5Bvalue%5D=&fields%5Bname-last%5D%5Btype%5D=string&fields%5Bname-last%5D%5Bvalue%5D=&fields%5Baddress-street-1%5D%5Btype%5D=string&fields%5Baddress-street-1%5D%5Bvalue%5D=&fields%5Baddress-street-2%5D%5Btype%5D=string&fields%5Baddress-street-2%5D%5Bvalue%5D=&fields%5Baddress-city%5D%5Btype%5D=string&fields%5Baddress-city%5D%5Bvalue%5D=&fields%5Baddress-subdivision%5D%5Btype%5D=string&fields%5Baddress-subdivision%5D%5Bvalue%5D=&fields%5Baddress-postal-code%5D%5Btype%5D=string&fields%5Baddress-postal-code%5D%5Bvalue%5D=&fields%5Baddress-country-code%5D%5Btype%5D=string&fields%5Baddress-country-code%5D%5Bvalue%5D=&fields%5Bbirthdate%5D%5Btype%5D=date&fields%5Bbirthdate%5D%5Bvalue%5D=&fields%5Bemail-address%5D%5Btype%5D=string&fields%5Bemail-address%5D%5Bvalue%5D=&fields%5Bphone-number%5D%5Btype%5D=string&fields%5Bphone-number%5D%5Bvalue%5D=%2B18043320874&fields%5Bidentification-number%5D%5Btype%5D=string&fields%5Bidentification-number%5D%5Bvalue%5D=&fields%5Bidentification-class%5D%5Btype%5D=string&fields%5Bidentification-class%5D%5Bvalue%5D=&fields%5Bselected-country-code%5D%5Btype%5D=string&fields%5Bselected-country-code%5D%5Bvalue%5D=US&fields%5Bphone%5D%5Btype%5D=string&fields%5Bphone%5D%5Bvalue%5D=&select=WRegularSeasonDetailedResults.csv)   |
| WNCAATourneyCompactResults.csv    | Raw NCAA tournament results used to build matchup-level target data                                                                                                                      | [Local/Kaggle file](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data?inquiry-id=inq_nP8ZBe1wnwQZXjGDRx2xgjp4iY7Y&reference-id=25013496&subject=25013496&status=approved&fields%5Bname-first%5D%5Btype%5D=string&fields%5Bname-first%5D%5Bvalue%5D=&fields%5Bname-middle%5D%5Btype%5D=string&fields%5Bname-middle%5D%5Bvalue%5D=&fields%5Bname-last%5D%5Btype%5D=string&fields%5Bname-last%5D%5Bvalue%5D=&fields%5Baddress-street-1%5D%5Btype%5D=string&fields%5Baddress-street-1%5D%5Bvalue%5D=&fields%5Baddress-street-2%5D%5Btype%5D=string&fields%5Baddress-street-2%5D%5Bvalue%5D=&fields%5Baddress-city%5D%5Btype%5D=string&fields%5Baddress-city%5D%5Bvalue%5D=&fields%5Baddress-subdivision%5D%5Btype%5D=string&fields%5Baddress-subdivision%5D%5Bvalue%5D=&fields%5Baddress-postal-code%5D%5Btype%5D=string&fields%5Baddress-postal-code%5D%5Bvalue%5D=&fields%5Baddress-country-code%5D%5Btype%5D=string&fields%5Baddress-country-code%5D%5Bvalue%5D=&fields%5Bbirthdate%5D%5Btype%5D=date&fields%5Bbirthdate%5D%5Bvalue%5D=&fields%5Bemail-address%5D%5Btype%5D=string&fields%5Bemail-address%5D%5Bvalue%5D=&fields%5Bphone-number%5D%5Btype%5D=string&fields%5Bphone-number%5D%5Bvalue%5D=%2B18043320874&fields%5Bidentification-number%5D%5Btype%5D=string&fields%5Bidentification-number%5D%5Bvalue%5D=&fields%5Bidentification-class%5D%5Btype%5D=string&fields%5Bidentification-class%5D%5Bvalue%5D=&fields%5Bselected-country-code%5D%5Btype%5D=string&fields%5Bselected-country-code%5D%5Bvalue%5D=US&fields%5Bphone%5D%5Btype%5D=string&fields%5Bphone%5D%5Bvalue%5D=&select=WNCAATourneyCompactResults.csv)   |
| WNCAATourneySeeds.csv             | Raw tournament seed data used to create numeric seed features                                                                                                                            | [Local/Kaggle file](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data?inquiry-id=inq_nP8ZBe1wnwQZXjGDRx2xgjp4iY7Y&reference-id=25013496&subject=25013496&status=approved&fields%5Bname-first%5D%5Btype%5D=string&fields%5Bname-first%5D%5Bvalue%5D=&fields%5Bname-middle%5D%5Btype%5D=string&fields%5Bname-middle%5D%5Bvalue%5D=&fields%5Bname-last%5D%5Btype%5D=string&fields%5Bname-last%5D%5Bvalue%5D=&fields%5Baddress-street-1%5D%5Btype%5D=string&fields%5Baddress-street-1%5D%5Bvalue%5D=&fields%5Baddress-street-2%5D%5Btype%5D=string&fields%5Baddress-street-2%5D%5Bvalue%5D=&fields%5Baddress-city%5D%5Btype%5D=string&fields%5Baddress-city%5D%5Bvalue%5D=&fields%5Baddress-subdivision%5D%5Btype%5D=string&fields%5Baddress-subdivision%5D%5Bvalue%5D=&fields%5Baddress-postal-code%5D%5Btype%5D=string&fields%5Baddress-postal-code%5D%5Bvalue%5D=&fields%5Baddress-country-code%5D%5Btype%5D=string&fields%5Baddress-country-code%5D%5Bvalue%5D=&fields%5Bbirthdate%5D%5Btype%5D=date&fields%5Bbirthdate%5D%5Bvalue%5D=&fields%5Bemail-address%5D%5Btype%5D=string&fields%5Bemail-address%5D%5Bvalue%5D=&fields%5Bphone-number%5D%5Btype%5D=string&fields%5Bphone-number%5D%5Bvalue%5D=%2B18043320874&fields%5Bidentification-number%5D%5Btype%5D=string&fields%5Bidentification-number%5D%5Bvalue%5D=&fields%5Bidentification-class%5D%5Btype%5D=string&fields%5Bidentification-class%5D%5Bvalue%5D=&fields%5Bselected-country-code%5D%5Btype%5D=string&fields%5Bselected-country-code%5D%5Bvalue%5D=US&fields%5Bphone%5D%5Btype%5D=string&fields%5Bphone%5D%5Bvalue%5D=&select=WNCAATourneySeeds.csv)   |

Bias Identification:

Several forms of bias may affect this dataset. First, there is omitted variable bias, because important predictors of game outcomes such as injuries, roster changes, coaching decisions, travel fatigue, and matchup-specific tactics are not included. Second, there is selection bias, since the prediction target is based only on tournament games, which include stronger teams than the full population of NCAA teams. Third, there is aggregation bias, because regular season performance is summarized into averages and win percentage, which may hide game-to-game variability and differences in opponent strength.

There is also potential historical bias because data from different seasons may reflect changes in team quality, play style, or broader trends in women’s college basketball. As a result, patterns learned from past seasons may not transfer perfectly to later seasons.

Bias Mitigation:

I addressed bias in several ways. To reduce data leakage, I constructed features only from regular season data and used tournament outcomes only as the target variable. This ensures that the model is predicting games using information that would actually be available before the tournament game is played. I also used difference-based features such as ScoreDiff, WinPctDiff, and SeedDiff, which help normalize team comparisons and focus the model on relative strength rather than absolute values.

To reduce the risk of overfitting and improve generalizability, I split the data into training and testing sets so that model performance is evaluated on unseen data. I also compared a simpler interpretable model, logistic regression, with a more flexible model, random forest, to see whether more complexity meaningfully improved performance. Finally, I interpret the model outputs as probabilistic estimates rather than definitive predictions, which helps account for the uncertainty and incompleteness of the available data.

Rational for Critical Decisions:

One critical decision was defining the unit of analysis as a single tournament game matchup, because that directly matches the prediction task. Another important decision was building features from regular season performance only. This was necessary because using final tournament game statistics would introduce data leakage and make the predictions unrealistic.

I also chose to represent each game twice, once from each team’s perspective, so that the target variable Win becomes a balanced binary outcome. This makes the classification setup cleaner and avoids having the model learn from an imbalanced winner-only structure. Another key judgment call was engineering difference-based features instead of using only separate team statistics. I chose this because relative differences in scoring, win percentage, and seeding better reflect the competitive relationship between two teams and are easier for the model to interpret.

Finally, I selected logistic regression as the baseline model because it is simple, interpretable, and appropriate for binary classification. I also included random forest as a comparison model to test whether a more flexible nonlinear model would improve performance. In this case, logistic regression performed slightly better, which supports keeping the simpler model as the preferred choice.

## Metadata
Schema: 

Data:

Data Dictionary:

Data Dictionary Quantificatiopn of Uncertainty:
