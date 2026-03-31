# Data Model Predicts NCAA Women’s Tournament Outcomes with High Accuracy

## Hook

Can a simple data model outperform complex machine learning in predicting NCAA tournament games? New analysis shows that just a few key performance metrics can reliably predict game outcomes.


## Problem Statement

Predicting outcomes in NCAA basketball tournaments is challenging due to the high variability and pressure of single-elimination games. Fans and analysts often rely on rankings, intuition, or expert opinions, but these approaches lack consistency and transparency.

This project addresses the problem of predicting NCAA women’s basketball tournament outcomes using historical data. Specifically, it evaluates whether regular-season performance and tournament seeding can be used to accurately predict which team will win a matchup.

## Solution Description

This project builds a data pipeline that transforms raw NCAA data into a structured relational dataset and applies machine learning models to predict tournament outcomes.

Key features include:
- Score differential between teams  
- Win percentage difference  
- Tournament seed rankings  

Two models were tested: logistic regression and random forest. The logistic regression model achieved an AUC of **0.828**, outperforming the more complex random forest model (AUC **0.807**).

These results demonstrate that simple, interpretable features are highly effective predictors of tournament success and do not require complex modeling techniques.

## Chart

![ROC Curve](../pipeline/project1_pipeline_files/roc_curve_comparison.png)

The ROC curve shows that logistic regression consistently outperforms random forest across classification thresholds, indicating stronger overall predictive performance.

## Why It Matters

This project highlights the power of interpretable data science. Rather than relying on complex models, analysts can use a small number of meaningful features to understand and predict tournament outcomes.

These findings are useful for sports analysts, teams, and fans seeking a more data-driven approach to understanding competitive performance.
