# DS 4320 Project 1: Predicting NCAA Women’s Basketball Tournament Outcomes

**Executive Summary:**

This project predicts outcomes of women’s NCAA tournament basketball games using regular season performance metrics and tournament seeding. By combining historical game data with engineered features such as scoring differentials and win percentage, the model identifies key factors that influence tournament success. The goal is to demonstrate how data-driven insights can support competitive strategy and decision-making in sports.

**Alka Link - eju2pk**

| Links |
|-------|
| [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19342879.svg)](https://doi.org/10.5281/zenodo.19342879) |
| [Press Release](https://github.com/alkalink1/databydesignproject1/tree/main/press%20release) |
| [Data](https://myuva-my.sharepoint.com/:f:/g/personal/eju2pk_virginia_edu/IgDeF5nA-Gn-T7l6UznITH87ARMYPEYqENMb4cyOPstwc9Y?e=oFiBZo) |
| [Pipeline](https://github.com/alkalink1/databydesignproject1/tree/main/pipeline) |
| [MIT License](https://github.com/alkalink1/databydesignproject1/blob/main/LICENSE) |

## Problem Definition
**General Problem:**

Predicting sports games outcomes

**Specific Problem Statement:**

The refined problem is to predict the outcome of NCAA college basketball games using historical game results and team performance statistics. By analyzing patterns in past games, like scoring margins, win-loss records, and game location, we aim to estimate the probability that a team will win a given matchup. The project will use historical NCAA basketball data, including regular season and tournament results, to explore how statistical patterns can be used to predict future game outcomes.


**Rationale for Project Refinement:**

The general problem of predicting sports game outcomes is extremely broad because many factors can influence the result of a game, including player injuries, team strategies, coaching decisions, and random variation. To make the problem more manageable, the project focuses specifically on NCAA college basketball games and uses structured historical data to model outcomes. This refinement allows the analysis to focus on measurable variables such as scoring margins, game location, and historical performance trends. By narrowing the problem to college basketball outcomes and using historical game statistics, the project becomes a clearly defined data analysis problem that can be explored using statistical and machine learning methods.


**Motivation:**

Predicting sports game outcomes is an important application of sports analytics. Teams, analysts, and fans increasingly rely on statistical models to evaluate team performance and estimate the probability of winning future games. In college basketball, predictive models are widely used during the NCAA tournament, where analysts attempt to forecast game outcomes for brackets and tournament simulations. By analyzing historical game data, analysts can identify patterns that help explain why some teams perform better than others. Developing predictive models for game outcomes can provide insights into the factors that most strongly influence winning and demonstrate how data can be used to support decision-making in sports.


[What Really Predicts Winning in March Madness?](https://github.com/alkalink1/databydesignproject1/blob/main/press%20release/press_release.md)

## Domain Exposition
**Terminology:**

| Term                 | Definition                                                                          |
| -------------------- | ----------------------------------------------------------------------------------- |
| Sports Analytics     | The use of data analysis and statistics to evaluate sports performance and outcomes |
| Win Probability      | The estimated likelihood that a team will win a game                                |
| Point Differential   | The difference between points scored and points allowed in a game                   |
| Home Court Advantage | The performance advantage teams often have when playing at home                     |
| Seed                 | A ranking assigned to teams in tournament brackets                                  |
| Predictive Model     | A statistical or machine learning model used to estimate future outcomes            |

**Domain:**

This project operates within the field of sports analytics, which focuses on using data to better understand athletic performance and predict future outcomes. In college basketball, large amounts of historical data are available describing game results, team statistics, and tournament performance. Analysts use these datasets to evaluate team strength, identify performance trends, and estimate the probability that a team will win a future game. Predictive modeling is widely used in this domain, especially during the NCAA tournament, where analysts and fans attempt to forecast game results and complete tournament brackets. By analyzing historical NCAA basketball game data, this project explores how statistical patterns in past games can help predict the outcomes of future matchups.

[Background Reading](https://myuva-my.sharepoint.com/:f:/g/personal/eju2pk_virginia_edu/IgDPtUFJxc8aSbEX77RQAcMQAfx3tmhBEDm67SGjhpOz4pI?e=Nffql1)

**Table of Readings:** 
| Title                                                                                     | Description                                                                                                              | File                                                                 |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| Kaggle March Machine Learning Mania Dataset                                               | Documentation describing the NCAA basketball dataset used for predictive modeling and tournament prediction competitions | [kaggle_march_madness_dataset.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/eju2pk_virginia_edu/IQC81MKhNI1sTrOm6tvceWM-ARoTXA7st7AAWRBFywn1B8g?e=CNjn5V)        |
| A Logistic Regression/Markov Chain Model for NCAA Basketball                              | Research paper describing statistical models used to predict NCAA basketball game outcomes                               | [logistic_regression_markov_ncaa.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/eju2pk_virginia_edu/IQB3ooTxn0vUTJ7wbZHcskYsAZbG2mm6p7M9hIhCXCPV3VY?e=pHsvmG)     |
| The Application of Machine Learning Techniques for Predicting Match Results in Team Sport | Review paper explaining how machine learning models can be used to predict sports game outcomes                          | [machine_learning_sports_prediction.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/eju2pk_virginia_edu/IQA2vQEbtPTXQ4a8uwa6Qoz9Aa54SZL-q-Vua-6uVtniptw?e=vWa8g5)  |
| On the Probability of Winning a Basketball Game                                           | Statistical analysis exploring how scoring patterns affect the probability of winning a basketball game                  | [probability_winning_basketball_game.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/eju2pk_virginia_edu/IQDeF4dSHjT5RIaRR98__yUfAWwNBEtLpsykEeoIdRu5lQQ?e=AzcNEU) |
| Predictive Modeling for Sports and Gaming                                                 | Research discussing how predictive models can be applied to sports analytics and outcome prediction                      | [predictive_modeling_sports.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/eju2pk_virginia_edu/IQDe69u4wMeRQrORB3UWLA23Afbi91SI_PMaPSE6DBrigEs?e=p2iEm8)          |

## Data Creation

**Provenance:** 

The dataset is constructed from publicly available NCAA women’s basketball data obtained via Kaggle. The raw data includes four primary files: regular season detailed results, tournament results, tournament seeds, and team identifiers. These datasets provide both game-level and team-level information necessary for modeling tournament outcomes.

The data acquisition process involved downloading CSV files from Kaggle and loading them into Python using pandas. No web scraping or manual data entry was performed, ensuring reproducibility and consistency.

**Code table:** 

| File | Description | Link |
|------|-------------|------|
| project1_pipeline.ipynb | Loads raw data, validates structure, constructs relational tables, and performs feature engineering | [project1_pipeline.ipynb](https://github.com/alkalink1/DS4320-Project1/blob/main/pipeline/project1_pipeline.ipynb) |
| project1_pipeline.md | Markdown version of the full pipeline notebook | [project1_pipeline.md](https://github.com/alkalink1/DS4320-Project1/blob/main/pipeline/project1_pipeline.md) |

**Bias Identification:**

Bias may be introduced in the data collection process due to limitations in the raw NCAA datasets. The data does not include contextual variables such as injuries, player-level performance, or strength of schedule. Additionally, tournament seeding reflects subjective committee decisions, which may introduce systematic bias.

**Bias Mitigation:**

To mitigate these biases, the analysis uses multiple features that capture different aspects of team performance, including scoring, win percentage, and seeding. Relative features (e.g., differences between teams) are used instead of absolute values to reduce the impact of scale differences across teams.

Additionally, the dataset is restricted to seasons where all source data is available to avoid bias caused by missing values.

**Rationale for Critical Decisions:**

Several key design decisions were made in constructing the dataset:

- A relational data structure was used to separate entities (teams, seasons, performance, and games), improving clarity and enabling SQL-based joins  
- Seasons were restricted to overlapping years across datasets to prevent missing data and ensure valid joins  
- Teams were randomly assigned as Team1 or Team2 in tournament games to avoid introducing bias toward winning teams  
- Feature engineering focused on interpretable variables (score differential, win percentage differential, seed differential) to support explainability  

These decisions were made to balance data completeness, interpretability, and model performance.

## Metadata
**Schema:** 

![ERD](/pipeline/project1_pipeline_files/erd.png)

The dataset is organized using a relational schema consisting of four tables: teams, seasons, regular_season_team_stats, and tournament_games.

The `teams` table stores team identifiers and names, while the `seasons` table defines the time dimension of the dataset. The `regular_season_team_stats` table contains aggregated performance metrics for each team within a season, including scoring, wins, and win percentage. The `tournament_games` table represents matchup-level data, where each row corresponds to a single NCAA tournament game.

Relationships are defined through shared keys. Each team appears in multiple records across both the regular season and tournament tables, and each season links performance and game outcomes. The relational structure enables efficient joins using DuckDB, allowing the pipeline to combine team performance and tournament data into a final modeling dataset.

**Data:**

| Table Name | Description | Link |
|------------|------------|------|
| teams | Contains team identifiers and names | [teams.csv](https://myuva-my.sharepoint.com/:x:/r/personal/eju2pk_virginia_edu/Documents/ds4320project1/data/teams.csv?d=w258e618ac386462790d0b371a91049ac&csf=1&web=1&e=drLWjo) |
| seasons | Contains the list of seasons used in the dataset | [seasons.csv](https://myuva-my.sharepoint.com/:x:/r/personal/eju2pk_virginia_edu/Documents/ds4320project1/data/seasons.csv?d=w45d1d70fec544aaeb3d0b4d1344a9756&csf=1&web=1&e=Tf2wn1) |
| regular_season_team_stats | Aggregated team performance metrics for each season | [regular_season_team_stats.csv](https://myuva-my.sharepoint.com/:x:/r/personal/eju2pk_virginia_edu/Documents/ds4320project1/data/regular_season_team_stats.csv?d=wb26b99e14959457da7ceda7fce3319e5&csf=1&web=1&e=Syd34e)|
| tournament_games | Tournament matchups including seeds and outcomes | [tournament_games.csv](https://myuva-my.sharepoint.com/:x:/r/personal/eju2pk_virginia_edu/Documents/ds4320project1/data/tournament_games.csv?d=wfc86006bc3654133ba4dd997d813bcbe&csf=1&web=1&e=0wOeD2) |
| final_model_dataset | Final modeling dataset created from SQL joins and feature engineering | [final_model_dataset.csv](https://myuva-my.sharepoint.com/:x:/r/personal/eju2pk_virginia_edu/Documents/ds4320project1/data/final_model_dataset.csv?d=w4cfc1d9a5819430fabf63d3b2c576f70&csf=1&web=1&e=4q82VR) |

**Data Dictionary:**

| Name | Data Type | Description | Example |
|------|----------|------------|---------|
| TeamID | integer | Unique team identifier | 3124 |
| TeamName | string | Name of the team | South Carolina |
| Season | integer | NCAA season year | 2023 |
| AvgScore | float | Average points scored per game | 74.5 |
| GamesPlayed | integer | Number of games played | 31 |
| Wins | integer | Number of wins | 27 |
| Losses | integer | Number of losses | 4 |
| WinPct | float | Win percentage | 0.871 |
| GameID | integer | Unique game identifier | 1 |
| Team1ID | integer | First team in matchup | 3124 |
| Team2ID | integer | Second team in matchup | 3177 |
| Team1Seed | integer | Tournament seed for Team 1 | 1 |
| Team2Seed | integer | Tournament seed for Team 2 | 8 |
| Team1Win | integer | Outcome variable (1 = win, 0 = loss) | 1 |

**Data Dictionary Quantification of Uncertainty:**

| Feature | Uncertainty |
|--------|------------|
| AvgScore | Moderate uncertainty due to variation in team performance across games |
| WinPct | Moderate uncertainty due to differences in strength of schedule |
| GamesPlayed | Low uncertainty (direct count of games) |
| Wins / Losses | Low uncertainty (observed outcomes) |
| Team1Seed / Team2Seed | Low uncertainty (official tournament rankings) |
| Team1Win | Low uncertainty (observed game outcome) |

Overall, the dataset has moderate uncertainty due to the absence of contextual variables such as injuries, player-level performance, and team dynamics.
