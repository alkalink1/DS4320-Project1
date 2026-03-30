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

**Provenance:** 

For this project, I used the Kaggle Women’s NCAA Basketball dataset to build a predictive dataset for tournament game outcomes. My project goal is to predict whether a team wins an NCAA tournament game using historical performance data, so I selected files that contain regular season results, tournament results, and tournament seed information. Specifically, I used WRegularSeasonDetailedResults.csv for regular season team performance, WNCAATourneyCompactResults.csv for tournament matchups and outcomes, and WNCAATourneySeeds.csv for tournament seed information. This dataset choice matches the project objective of predicting NCAA basketball game outcomes using historical team statistics.

To create the final modeling dataset, I first computed team-level regular season statistics by season. I calculated each team’s average score using both games they won and games they lost, then combined those into a single average scoring feature. I also calculated each team’s win percentage from regular season wins and losses. Next, I extracted each team’s tournament seed and converted the seed code into a numeric seed value. After that, I created a matchup-level tournament dataset by taking each NCAA tournament game and representing it from both team perspectives: one row where Team1 is the winner and one row where Team1 is the loser. This created a balanced binary outcome variable called Win. Finally, I merged team-level features into each matchup for Team1 and Team2 and engineered difference-based features (ScoreDiff, WinPctDiff, and SeedDiff) so the model could learn relative team strength rather than raw team values. The final dataset is therefore game-level, prediction-oriented, and constructed entirely from pre-game information, avoiding data leakage.


**Code to create data:** 

| File Name                         | Description                                                                                                                                                                              | Link                |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| data_pipeline_and_model.ipynb     | Loads Kaggle NCAA datasets, creates team-level regular season features, merges them into tournament matchups, engineers difference variables, and trains/evaluates classification models | [Colab notebook link](https://colab.research.google.com/drive/1wo1f_aWnYOD9oRVbx0zrrs_nh9Ayw6Yd?usp=sharing) |
| WRegularSeasonDetailedResults.csv | Raw regular season game-level results used to compute average score and win percentage                                                                                                   | [Local/Kaggle file](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data?inquiry-id=inq_nP8ZBe1wnwQZXjGDRx2xgjp4iY7Y&reference-id=25013496&subject=25013496&status=approved&fields%5Bname-first%5D%5Btype%5D=string&fields%5Bname-first%5D%5Bvalue%5D=&fields%5Bname-middle%5D%5Btype%5D=string&fields%5Bname-middle%5D%5Bvalue%5D=&fields%5Bname-last%5D%5Btype%5D=string&fields%5Bname-last%5D%5Bvalue%5D=&fields%5Baddress-street-1%5D%5Btype%5D=string&fields%5Baddress-street-1%5D%5Bvalue%5D=&fields%5Baddress-street-2%5D%5Btype%5D=string&fields%5Baddress-street-2%5D%5Bvalue%5D=&fields%5Baddress-city%5D%5Btype%5D=string&fields%5Baddress-city%5D%5Bvalue%5D=&fields%5Baddress-subdivision%5D%5Btype%5D=string&fields%5Baddress-subdivision%5D%5Bvalue%5D=&fields%5Baddress-postal-code%5D%5Btype%5D=string&fields%5Baddress-postal-code%5D%5Bvalue%5D=&fields%5Baddress-country-code%5D%5Btype%5D=string&fields%5Baddress-country-code%5D%5Bvalue%5D=&fields%5Bbirthdate%5D%5Btype%5D=date&fields%5Bbirthdate%5D%5Bvalue%5D=&fields%5Bemail-address%5D%5Btype%5D=string&fields%5Bemail-address%5D%5Bvalue%5D=&fields%5Bphone-number%5D%5Btype%5D=string&fields%5Bphone-number%5D%5Bvalue%5D=%2B18043320874&fields%5Bidentification-number%5D%5Btype%5D=string&fields%5Bidentification-number%5D%5Bvalue%5D=&fields%5Bidentification-class%5D%5Btype%5D=string&fields%5Bidentification-class%5D%5Bvalue%5D=&fields%5Bselected-country-code%5D%5Btype%5D=string&fields%5Bselected-country-code%5D%5Bvalue%5D=US&fields%5Bphone%5D%5Btype%5D=string&fields%5Bphone%5D%5Bvalue%5D=&select=WRegularSeasonDetailedResults.csv)   |
| WNCAATourneyCompactResults.csv    | Raw NCAA tournament results used to build matchup-level target data                                                                                                                      | [Local/Kaggle file](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data?inquiry-id=inq_nP8ZBe1wnwQZXjGDRx2xgjp4iY7Y&reference-id=25013496&subject=25013496&status=approved&fields%5Bname-first%5D%5Btype%5D=string&fields%5Bname-first%5D%5Bvalue%5D=&fields%5Bname-middle%5D%5Btype%5D=string&fields%5Bname-middle%5D%5Bvalue%5D=&fields%5Bname-last%5D%5Btype%5D=string&fields%5Bname-last%5D%5Bvalue%5D=&fields%5Baddress-street-1%5D%5Btype%5D=string&fields%5Baddress-street-1%5D%5Bvalue%5D=&fields%5Baddress-street-2%5D%5Btype%5D=string&fields%5Baddress-street-2%5D%5Bvalue%5D=&fields%5Baddress-city%5D%5Btype%5D=string&fields%5Baddress-city%5D%5Bvalue%5D=&fields%5Baddress-subdivision%5D%5Btype%5D=string&fields%5Baddress-subdivision%5D%5Bvalue%5D=&fields%5Baddress-postal-code%5D%5Btype%5D=string&fields%5Baddress-postal-code%5D%5Bvalue%5D=&fields%5Baddress-country-code%5D%5Btype%5D=string&fields%5Baddress-country-code%5D%5Bvalue%5D=&fields%5Bbirthdate%5D%5Btype%5D=date&fields%5Bbirthdate%5D%5Bvalue%5D=&fields%5Bemail-address%5D%5Btype%5D=string&fields%5Bemail-address%5D%5Bvalue%5D=&fields%5Bphone-number%5D%5Btype%5D=string&fields%5Bphone-number%5D%5Bvalue%5D=%2B18043320874&fields%5Bidentification-number%5D%5Btype%5D=string&fields%5Bidentification-number%5D%5Bvalue%5D=&fields%5Bidentification-class%5D%5Btype%5D=string&fields%5Bidentification-class%5D%5Bvalue%5D=&fields%5Bselected-country-code%5D%5Btype%5D=string&fields%5Bselected-country-code%5D%5Bvalue%5D=US&fields%5Bphone%5D%5Btype%5D=string&fields%5Bphone%5D%5Bvalue%5D=&select=WNCAATourneyCompactResults.csv)   |
| WNCAATourneySeeds.csv             | Raw tournament seed data used to create numeric seed features                                                                                                                            | [Local/Kaggle file](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data?inquiry-id=inq_nP8ZBe1wnwQZXjGDRx2xgjp4iY7Y&reference-id=25013496&subject=25013496&status=approved&fields%5Bname-first%5D%5Btype%5D=string&fields%5Bname-first%5D%5Bvalue%5D=&fields%5Bname-middle%5D%5Btype%5D=string&fields%5Bname-middle%5D%5Bvalue%5D=&fields%5Bname-last%5D%5Btype%5D=string&fields%5Bname-last%5D%5Bvalue%5D=&fields%5Baddress-street-1%5D%5Btype%5D=string&fields%5Baddress-street-1%5D%5Bvalue%5D=&fields%5Baddress-street-2%5D%5Btype%5D=string&fields%5Baddress-street-2%5D%5Bvalue%5D=&fields%5Baddress-city%5D%5Btype%5D=string&fields%5Baddress-city%5D%5Bvalue%5D=&fields%5Baddress-subdivision%5D%5Btype%5D=string&fields%5Baddress-subdivision%5D%5Bvalue%5D=&fields%5Baddress-postal-code%5D%5Btype%5D=string&fields%5Baddress-postal-code%5D%5Bvalue%5D=&fields%5Baddress-country-code%5D%5Btype%5D=string&fields%5Baddress-country-code%5D%5Bvalue%5D=&fields%5Bbirthdate%5D%5Btype%5D=date&fields%5Bbirthdate%5D%5Bvalue%5D=&fields%5Bemail-address%5D%5Btype%5D=string&fields%5Bemail-address%5D%5Bvalue%5D=&fields%5Bphone-number%5D%5Btype%5D=string&fields%5Bphone-number%5D%5Bvalue%5D=%2B18043320874&fields%5Bidentification-number%5D%5Btype%5D=string&fields%5Bidentification-number%5D%5Bvalue%5D=&fields%5Bidentification-class%5D%5Btype%5D=string&fields%5Bidentification-class%5D%5Bvalue%5D=&fields%5Bselected-country-code%5D%5Btype%5D=string&fields%5Bselected-country-code%5D%5Bvalue%5D=US&fields%5Bphone%5D%5Btype%5D=string&fields%5Bphone%5D%5Bvalue%5D=&select=WNCAATourneySeeds.csv)   |

**Bias Identification:**

Several forms of bias may affect this dataset. First, there is omitted variable bias, because important predictors of game outcomes such as injuries, roster changes, coaching decisions, travel fatigue, and matchup-specific tactics are not included. Second, there is selection bias, since the prediction target is based only on tournament games, which include stronger teams than the full population of NCAA teams. Third, there is aggregation bias, because regular season performance is summarized into averages and win percentage, which may hide game-to-game variability and differences in opponent strength.

There is also potential historical bias because data from different seasons may reflect changes in team quality, play style, or broader trends in women’s college basketball. As a result, patterns learned from past seasons may not transfer perfectly to later seasons.

**Bias Mitigation:**


**Rational for Critical Decisions:**

## Metadata
**Schema:** 
![image.png](<img width="378" height="378" alt="image" src="https://github.com/user-attachments/assets/bffcdd52-2f48-4653-a06c-ea175e77adb3" />)

**Data:**

| Table Name     | Description                                                                                                                                | Link                                 |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| NCAA_Game_Data | Final game-level modeling dataset containing tournament matchups, merged regular season team features, and engineered difference variables | [Generated in notebook / exported CSV](https://drive.google.com/file/d/1STwlRCdWWx-Uyg34DcEQGQKepDIG_AKQ/view?usp=sharing) |

**Data Dictionary:**

| Name       | Data Type     | Description                                     | Example |
| ---------- | ------------- | ----------------------------------------------- | ------- |
| Season     | integer       | NCAA season year                                | 2022    |
| Team1      | integer       | Team ID for the first team in the matchup       | 3101    |
| Team2      | integer       | Team ID for the second team in the matchup      | 3376    |
| AvgScore_1 | float         | Team1 average regular season score              | 74.3    |
| AvgScore_2 | float         | Team2 average regular season score              | 68.9    |
| Wins_1     | float/integer | Number of Team1 regular season wins             | 24      |
| Losses_1   | float/integer | Number of Team1 regular season losses           | 6       |
| Games_1    | float/integer | Number of Team1 regular season games            | 30      |
| WinPct_1   | float         | Team1 regular season win percentage             | 0.800   |
| SeedNum_1  | float/integer | Team1 numeric NCAA tournament seed              | 3       |
| Wins_2     | float/integer | Number of Team2 regular season wins             | 20      |
| Losses_2   | float/integer | Number of Team2 regular season losses           | 10      |
| Games_2    | float/integer | Number of Team2 regular season games            | 30      |
| WinPct_2   | float         | Team2 regular season win percentage             | 0.667   |
| SeedNum_2  | float/integer | Team2 numeric NCAA tournament seed              | 8       |
| ScoreDiff  | float         | Difference in average score: Team1 minus Team2  | 5.4     |
| WinPctDiff | float         | Difference in win percentage: Team1 minus Team2 | 0.133   |
| SeedDiff   | float         | Difference in seed number: Team1 minus Team2    | -5      |
| Win        | binary        | Target variable: 1 if Team1 wins, 0 otherwise   | 1       |

**Data Dictionary Quantificatiopn of Uncertainty:**

| Feature                 | Uncertainty                                                                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| AvgScore_1 / AvgScore_2 | Moderate uncertainty; averages summarize many games and may hide variation due to opponent strength or game context  |
| Wins / Losses / Games   | Low uncertainty; directly counted from recorded regular season results                                               |
| WinPct_1 / WinPct_2     | Moderate uncertainty; reliable as a summary, but influenced by schedule difficulty and conference strength           |
| SeedNum_1 / SeedNum_2   | Low uncertainty; directly derived from official tournament seed assignments                                          |
| ScoreDiff               | Moderate uncertainty; derived from average scoring and therefore sensitive to the uncertainty in average score       |
| WinPctDiff              | Moderate uncertainty; derived from win percentages and reflects summary-level rather than game-specific team quality |
| SeedDiff                | Low uncertainty; derived from official seed values                                                                   |
| Win                     | Low uncertainty; directly based on recorded tournament game outcomes                                                 |

