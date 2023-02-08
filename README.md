# btcpredict

BTCPREDICTAI.py

The given code is a Python script for performing regression analysis on a financial dataset. It employs 3 machine learning algorithms, namely Random Forest Regression, Gradient Boosting Regression and XGBoost Regression, to predict the closing price of a stock based on various financial features. The script starts by loading a CSV file containing the financial data using the pandas library and extracting some additional features from the data. The three regression models are then instantiated and their hyperparameters are defined for tuning using GridSearchCV. The data is split into training and test sets and the models are trained on the training data. A progress bar is used to track the training progress of each model. The models are then used to make predictions on the test data and the results are combined to form a final prediction. Finally, the results of each model and the final prediction are printed in terms of R2 score and mean absolute error.
