import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import tensorflow as tf
import seaborn as sns


# download open, high, low, close, volume data from api, outputsize = full (all data range)
def download_stock_df(stock_name) -> DataFrame:
    ts = TimeSeries(key='Z3QM56MJPC8SYUT7', output_format='pandas', indexing_type='date')
    stock_data_df, stock_meta_df = ts.get_daily_adjusted(stock_name, outputsize='full')
    stock_data_df.drop(['5. adjusted close', '7. dividend amount', '8. split coefficient'], axis=1, inplace=True)
    stock_data_df.columns = ["open", "high", "low", "close", "volume"]
    return stock_data_df.iloc[::-1]


# Filtreaza si scaleaza datele
def prepare_data(df: DataFrame):
    # Ce features vrem sa luam din dataset
    features = ['open', 'high', 'low', 'close', 'volume']

    print('FEATURE LIST')
    print([f for f in features])

    # Extragem doar colanele din lista noastra
    df_filter = df[features]

    np_filter_unscaled = np.array(df_filter)
    print(np_filter_unscaled.shape)

    np_c_unscaled = np.array(df['close']).reshape(-1, 1)

    # Scalam fiecare feature intr-un interval [0,1]
    scaler_train = MinMaxScaler()
    np_scaled = scaler_train.fit_transform(np_filter_unscaled)

    # Am creat un scaler separat pentru Y
    scaler_pred = MinMaxScaler()
    np_scaled_c = scaler_pred.fit_transform(np_c_unscaled)

    #return np_filter_unscaled, np_c_unscaled, scaler_train, scaler_pred
    return np_scaled, np_scaled_c, scaler_train, scaler_pred


# Formateaza datele in formatul acceptat de algoritm: [samples, time steps, features]
def partition_dataset(input_sequence_length, output_sequence_length, data, y_index):
    x_part, y_part = [], []
    data_len = data.shape[0]
    for i in range(input_sequence_length, data_len - output_sequence_length):
        x_part.append(data[i - input_sequence_length:i, :])
        y_part.append(data[i:i + output_sequence_length, y_index])
    return np.array(x_part), np.array(y_part)

# input shape = timesteps, features
# def LSTM_model(input_shape, no_outputs) -> Sequential:
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=50, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=50))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=no_outputs))
#     return model