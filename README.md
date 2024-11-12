This project aims to find which models (linear and nonlinear) provide the highest accuracy for predicting NFL winners and losers. 
The data used is from www.pro-football-reference.com

Methodology
The project applies various machine learning models to analyze and predict game outcomes. Key steps include:
Data Cleaning and Formatting: The raw data was pre-processed to handle inconsistencies, address missing values, and format it for model input. Along with shifting certain predictors like Record and Spread
Fixing the skewness of data with the help of the Python package yeojohnson 
Feature Engineering: Creating additional features like point differences and adjusting existing ones to improve model performance.
Handling Multicollinearity: Addressing correlated features using VIF to prevent skewed model results and ensure more accurate predictions.
Splitting the data and using folds
Model Training: Using models such as linear regression, logistic regression, and non-linear algorithms like random forests and gradient boosting.
Evaluation: Assessing models based on accuracy and other performance metrics to identify the best approach for predicting winners and losers.
Results: Initial trials resulted in models achieving up to 71% accuracy in predicting game outcomes. Further optimization and feature tuning are ongoing to improve this benchmark.

To complement the predictive analysis, an interactive website www... was developed to display player and team statistics with visual graphs. This tool provides users with a detailed breakdown of player performance and team stats for further exploration.
