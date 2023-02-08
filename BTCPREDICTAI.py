import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import xgboost as xgb
import progressbar

# Load the data
data = pd.read_csv("bitprice.csv")

# Extract additional features
data['moving_avg_7'] = data['close'].rolling(7).mean()
data['moving_avg_30'] = data['close'].rolling(30).mean()
data['price_diff'] = data['close'].diff()
data['volume_diff'] = data['Volume USD'].diff()
data.dropna(inplace=True)

# Train the models
rf = RandomForestRegressor()
gb = GradientBoostingRegressor()
xgb_model = xgb.XGBRegressor()

# Define the parameters for GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [2, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}

param_grid_gb = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [2, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'learning_rate': [0.1, 0.01, 0.001]
}

param_grid_xgb = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [2, 5, 7, 10],
    'learning_rate': [0.1, 0.01, 0.001]
}

# Use GridSearchCV to tune the hyperparameters
features = ['close', 'moving_avg_7', 'moving_avg_30', 'price_diff', 'Volume USD']
rf_grid = GridSearchCV(rf, param_grid_rf, cv=5)
gb_grid = GridSearchCV(gb, param_grid_gb, cv=5)
xgb_grid = GridSearchCV(xgb_model, param_grid_xgb, cv=5)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data['close'], test_size=0.2, random_state=42)

# Fit the models to the training data
bar = progressbar.ProgressBar(max_value=3)
for i in range(3):
    if i == 0:
        rf_grid.fit(X_train, y_train)
    elif i == 1:
        gb_grid.fit(X_train, y_train)
    else:
        xgb_grid.fit(X_train, y_train)
    bar.update(i)

# Use the models to make predictions on the test data
rf_pred = rf_grid.predict(X_test)
gb_pred = gb_grid.predict(X_test)
xgb_pred = xgb_grid.predict(X_test)

# Combine the predictions of multiple models and improve the accuracy of the prediction
final_pred = (rf_pred + gb_pred + xgb_pred) / 3

# Print the results of each model
print("Random Forest Regression Results:")
print("R2 score: ", rf_grid.score(X_test, y_test))
print("Mean Absolute Error: ", mean_absolute_error(y_test, rf_pred))

print("\nGradient Boosting Regression Results:")
print("R2 score: ", gb_grid.score(X_test, y_test))
print("Mean Absolute Error: ", mean_absolute_error(y_test, gb_pred))

print("\nXGBoost Regression Results:")
print("R2 score: ", xgb_grid.score(X_test, y_test))
print("Mean Absolute Error: ", mean_absolute_error(y_test, xgb_pred))

print("\nFinal Predicted Results:")
print("R2 score: ", r2_score(y_test, final_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, final_pred))
