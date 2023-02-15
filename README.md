# Bitcoin Price Predictor

This is a Python script that imports various libraries to build and train an LSTM model to predict the next hour's price of the BTCUSDT trading pair on Binance. The script collects data using the Binance API client, processes it using Pandas and NumPy, and then prepares it for training using the Scikit-learn library.

The LSTM model is defined using the Keras library and includes three LSTM layers, with each layer followed by a dropout layer to prevent overfitting. The model is compiled using the Adam optimizer and the mean squared error loss function.

After defining the hyperparameters to search over using the GridSearchCV class, the script fits the GridSearchCV object to the training data and prints the best hyperparameters found by the grid search. Then the best hyperparameters are used to build and train the LSTM model.

Finally, the trained model is used to predict the next hour's price, and the predicted price is printed to the console.

