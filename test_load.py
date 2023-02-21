import math

import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import tensorflow as tf
import seaborn as sns
from utils import download_stock_df, prepare_data, partition_dataset

data = download_stock_df("AAPL")
print(data)
index_Close = data.columns.get_loc("close")
np_scaled, np_scaled_c, scaler_train, scaler_pred = prepare_data(data)
model: tf.keras.Sequential = tf.keras.models.load_model('saved_model/best_model')

model.summary()

input_sequence_length = 30
output_sequence_length = 7

train_data_length = math.ceil(np_scaled.shape[0] * 0.8)

train_data = np_scaled[:train_data_length, :]
test_data = np_scaled[train_data_length - input_sequence_length:, :]

x_train, y_train = partition_dataset(input_sequence_length, output_sequence_length, train_data, index_Close)
x_test, y_test = partition_dataset(input_sequence_length, output_sequence_length, test_data, index_Close)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

y_pred_scaled = model.predict(x_test)
# Facem unscale
y_pred_unscaled = scaler_pred.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_pred.inverse_transform(y_test).reshape(-1, output_sequence_length)

# Mean Absolute Error (MAE)
MAE = tf.keras.metrics.mean_absolute_error(y_test_unscaled, y_pred_unscaled)
print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')


# # Mean Absolute Percentage Error (MAPE)
# MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred_unscaled) / y_test_unscaled))) * 100
# print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')
#
# # Median Absolute Percentage Error (MDAPE)
# MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred_unscaled) / y_test_unscaled))) * 100
# print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')


def prepare_df(i, x, y, y_pred_unscaled):
    # Preluam valorile preturilor din input
    x_test_unscaled_df = pd.DataFrame(scaler_train.inverse_transform(x[i])[:, index_Close]).rename(
        columns={0: 'x_test'})
    y_test_unscaled_df = pd.DataFrame(scaler_pred.inverse_transform(y)[i]).rename(columns={0: 'y_test'})
    y_pred_df = pd.DataFrame(y_pred_unscaled[i]).rename(columns={0: 'y_pred'})
    return x_test_unscaled_df, y_pred_df, y_test_unscaled_df


def plot_forecast(x_test_unscaled_df, y_test_unscaled_df, y_pred_df, title):
    df_merge = y_pred_df.join(y_test_unscaled_df, how='left')
    # print(df_merge)
    df_merge_ = pd.concat([x_test_unscaled_df, df_merge]).reset_index(drop=True)
    print(df_merge_)
    # Plot
    df_merge_.plot()
    plt.title(title)
    plt.grid()
    plt.show()


x_test_unscaled_df, y_pred_df, y_test_unscaled_df = prepare_df(y_test.shape[0] - 1, x_test, y_test, y_pred_unscaled)
plot_forecast(x_test_unscaled_df, y_test_unscaled_df, y_pred_df, "AAPL")
