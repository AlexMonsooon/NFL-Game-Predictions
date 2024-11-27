This project aims to find which models (linear and nonlinear) provide the highest accuracy for predicting NFL winners and losers. 
The data used is from www.pro-football-reference.com

Methodology
The project applies various machine learning models to analyze and predict game outcomes. Key steps include:
1. Data Cleaning and Formatting: The raw data was pre-processed to handle inconsistencies, address missing values, and format it for model input. Along with shifting certain predictors like Record and Spread
2. Fixing the skewness of data with the help of the Python package yeojohnson 
3. Feature Engineering: Creating additional features like point differences and adjusting existing ones to improve model performance.
4. Handling Multicollinearity: Addressing correlated features using VIF to prevent skewed model results and ensure more accurate predictions.
5. Splitting the data and using folds
6. Model Training: Using models such as linear regression, logistic regression, and non-linear algorithms like random forests and gradient boosting.
7. Evaluation: Assessing models based on accuracy and other performance metrics to identify the best approach for predicting winners and losers.
8. Results: Initial trials resulted in models achieving up to 71% accuracy in predicting game outcomes. Further optimization and feature tuning are ongoing to improve this benchmark.

To complement the predictive analysis, an interactive website https://proplayermetrics.com/ was developed to display player and team statistics with visual graphs. This tool provides users with a detailed breakdown of player performance and team stats for further exploration.
