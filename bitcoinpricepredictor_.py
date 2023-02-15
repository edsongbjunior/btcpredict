#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 00:28:36 2023

@author: edsongurgel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:09:04 2022

@author: edsongurgel
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# Import the Binance API Client
from binance.client import Client
# Define your Binance API keys
api_key = 'Your_Key'
api_secret = 'Your_Secret' 

# Create an instance of the Binance API Client
client = Client(api_key, api_secret)

# Define the parameters for the data collection
symbol = 'BTCUSDT'   # The symbol for the trading pair
interval = '1h'     # The interval for the data collection
limit = 1000        # The number of data points to collect

# Collect the data using the Binance API Client
klines = client.get_historical_klines(symbol, interval, f"{limit} hours ago UTC")

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Prepare the data for training the model
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']].values)
lookback = 48  # Use the previous 48 hours of data to predict the next hour's price
x_data = []
y_data = []
for i in range(lookback, len(data)):
    x_data.append(data[i-lookback:i, :])
    y_data.append(data[i, 3])  # Use the closing price as the target variable
x_data, y_data = np.array(x_data), np.array(y_data)

# Define the hyperparameters to search over
param_grid = {
    'units': [16, 32, 64],
    'dropout': [0.1, 0.2, 0.3],
    'batch_size': [32, 64, 128]
}

# Define the LSTM model
def create_model(units, dropout):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(x_data.shape[1], x_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create an instance of the KerasRegressor wrapper
model = KerasRegressor(build_fn=create_model, epochs=10)

# Create an instance of the GridSearchCV class
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(x_data, y_data)

# Print the best hyperparameters found by the grid search
print("Best Parameters: ", grid_search.best_params_)

# Use the best hyperparameters to build and train the LSTM model
best_units = grid_search.best_params_['units']
best_dropout = grid_search.best_params_['dropout']
best_batch_size = grid_search.best_params_['batch_size']
model = create_model(best_units, best_dropout)
model.fit(x_data, y_data, epochs=10, batch_size=best_batch_size)

# Use the trained model to predict the next hour's price
last_data = df.tail(lookback)[['open', 'high', 'low', 'close', 'volume']].values
last_data = scaler.transform(last_data)
x_test = np.array([last_data])
predicted_price = scaler.inverse_transform(model.predict(x_test))

# Calculate the percentage change in price over the last hour
last_price = df.tail(1)['close'].values[0]
percent_change = (predicted_price[0][0] - last_price) / last_price

# Print the predicted price and the buy, hold, or sell signal
if percent_change > 0.01:
 print(f"The predicted price for the next hour is: {predicted_price[0][0]:.2f}.\nBuy signal.")
elif percent_change < -0.01:
 print(f"The predicted price for the next hour is: {predicted_price[0][0]:.2f}.\nSell signal.")
else:
 print(f"The predicted price for the next hour is: {predicted_price[0][0]:.2f}.\nHold signal.")
