import math

import numpy
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Dropout, Dense, LSTM
from tensorflow.python.keras.losses import mean_absolute_error
import tensorflow as tf



# --------------------- FUNCTIONS -----------------------------

# download open, high, low, close, volume data from api, outputsize = full (all data range)
def download_stock_df(stock_name) -> DataFrame:
    ts = TimeSeries(key='Z3QM56MJPC8SYUT7', output_format='pandas', indexing_type='date')
    stock_data_df, stock_meta_df = ts.get_daily_adjusted(stock_name, outputsize='full')
    stock_data_df.drop(['5. adjusted close', '7. dividend amount', '8. split coefficient'], axis=1, inplace=True)
    stock_data_df.columns = ["open", "high", "low", "close", "volume"]
    return stock_data_df


# prepares the x, y data and scales it using MinMax
def prepare_data(df: DataFrame):
    # List of considered Features
    FEATURES = ['open', 'high', 'low', 'close', 'volume']

    print('FEATURE LIST')
    print([f for f in FEATURES])

    # Create the dataset with features and filter the data to the list of FEATURES
    df_filter = df[FEATURES]

    # Convert the data to numpy values
    np_filter_unscaled = np.array(df_filter)
    print(np_filter_unscaled.shape)

    np_c_unscaled = np.array(df['close']).reshape(-1, 1)

    # Creating a separate scaler that works on a single column for scaling predictions
    # Scale each feature to a range between 0 and 1
    scaler_train = MinMaxScaler()
    np_scaled = scaler_train.fit_transform(np_filter_unscaled)

    # Create a separate scaler for a single column
    scaler_pred = MinMaxScaler()
    np_scaled_c = scaler_pred.fit_transform(np_c_unscaled)

    return np_scaled, np_scaled_c, scaler_train, scaler_pred


# The LSTM needs data with the format of [samples, time steps, features]
# Here, we create N samples, input_sequence_length time steps per sample, and f features
def partition_dataset(input_sequence_length, output_sequence_length, data, y_index):
    x_part, y_part = [], []
    data_len = data.shape[0]
    for i in range(input_sequence_length, data_len - output_sequence_length):
        x_part.append(data[i - input_sequence_length:i,
                      :])
        y_part.append(data[i:i + output_sequence_length,
                      y_index])
    x_part = np.array(x_part)
    y_part = np.array(y_part)
    return x_part, y_part


# input shape = timesteps, features
def LSTM_model(input_shape, no_outputs) -> Sequential:
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=no_outputs))
    return model


# ------------------ Starting script --------------------

data = download_stock_df("AAPL")
index_Close = data.columns.get_loc("close")
np_scaled, np_scaled_c, scaler_train, scaler_pred = prepare_data(data)

# Set the input_sequence_length length - this is the timeframe used to make a single prediction
input_sequence_length = 30
# The output sequence length is the number of steps that the neural network predicts
output_sequence_length = 7  #

# Split the training data into train and train data sets
# As a first step, we get the number of rows to train the model on 80% of the data
train_data_length = math.ceil(np_scaled.shape[0] * 0.8)

# Create the training and test data
train_data = np_scaled[:train_data_length, :]
test_data = np_scaled[train_data_length - input_sequence_length:, :]

# Generate training data and test data
x_train, y_train = partition_dataset(input_sequence_length, output_sequence_length, train_data, index_Close)
x_test, y_test = partition_dataset(input_sequence_length, output_sequence_length, test_data, index_Close)

# Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = LSTM_model((x_train.shape[1], x_train.shape[2]), y_train.shape[1])
# model.summary()
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
# learning_rate_arr = [0.001, 0.01, 0.1, 0.2, 0.3]
# batches_array = [10, ]
# epochs_array = [5, 10, 20, 35, 50, 75, 100, 120, 150]



# Training the model
# epochs = 40
# batch_size = 64
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# fit model
history = model.fit(x_train,
                    y_train,
                    validation_split=0.3,
                    epochs=40,
                    batch_size=64,
                    callbacks=[es, mc])


results = model.evaluate(x_test, y_test)
print("test loss, test acc:", np.round(results, 4))
# Plot training & validation loss values
# fig, ax = plt.subplots(figsize=(10, 5), sharex="all")
# plt.plot(history.history["loss"])
# plt.title("Model loss")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
# plt.legend(["Train", "Test"], loc="upper left")
# plt.grid()
# plt.show()

# -------------------------- TESTARE ----------------------------

# Get the predicted values
# y_pred_scaled = model.predict(x_test)
#
# # Unscale the predicted values
# y_pred_unscaled = scaler_pred.inverse_transform(y_pred_scaled)
# y_test_unscaled = scaler_pred.inverse_transform(y_test).reshape(-1, output_sequence_length)
#
# # Mean Absolute Error (MAE)
# MAE = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
# print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')
#
# # Mean Absolute Percentage Error (MAPE)
# MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred_unscaled) / y_test_unscaled))) * 100
# print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')
#
# # Median Absolute Percentage Error (MDAPE)
# MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred_unscaled) / y_test_unscaled))) * 100
# print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

# --------------- GRAFIC PREDICTIE ----------------------------


# def plot_test_batch(x_test: ndarray, y_test: ndarray, y_pred: ndarray, batch_no):
#     plt.plot
