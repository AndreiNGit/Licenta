import math
import sys
import numpy as np
from keras_tuner import HyperParameters
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from utils import download_stock_df, prepare_data, partition_dataset

stock_symbol = "AAPL"#sys.argv[1]
data = download_stock_df(stock_symbol)
index_Close = data.columns.get_loc("close")
np_scaled, np_scaled_c, scaler_train, scaler_pred = prepare_data(data)

# input_sequence_length length - timeframe folosit pentru predictie
input_sequence_length = 30
# output_sequence_length - numarul de zile prezise de algoritm
output_sequence_length = 7

train_data_length = math.ceil(np_scaled.shape[0] * 0.8)

# Imparte setul de date in training si test
train_data = np_scaled[:train_data_length, :]
test_data = np_scaled[train_data_length - input_sequence_length:, :]

# Impartim setul de date in x si y formatate pentru adaugarea in algoritm
x_train, y_train = partition_dataset(input_sequence_length, output_sequence_length, np_scaled, index_Close)
x_test, y_test = partition_dataset(input_sequence_length, output_sequence_length, np_scaled, index_Close)


def define_model(layers, units, dropout, lr, dense):
    model = tf.keras.Sequential()
    model.add(keras.layers.LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    for i in range(layers):
        model.add(keras.layers.LSTM(units=units, return_sequences=True))
        if dropout:
            model.add(keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0, max_value=0.5, step=0.1)))
    model.add(keras.layers.LSTM(units=units))
    if dense:
        model.add(keras.layers.Dense(units=hp.Int("dense_units", min_value=10, max_value=100, step=10)))
    model.add(keras.layers.Dense(units=7))
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[keras.metrics.RootMeanSquaredError(name='rmse'),
                           keras.metrics.MeanAbsoluteError(name='mae')]
                  )
    return model


hp = HyperParameters()


def model_builder(hp):
    layers = hp.Int("layers", min_value=0, max_value=3, step=1)
    units = hp.Int("units", min_value=10, max_value=100, step=10)
    dense = hp.Boolean("dense")
    dropout = hp.Boolean("dropout")
    lr = hp.Choice("lr", values=[1e-2, 1e-3])

    model = define_model(
        layers=layers, units=units, dropout=dropout, lr=lr, dense=dense,
    )
    return model


es = tf.keras.callbacks.EarlyStopping(monitor='val_rmse', mode='min', verbose=1, patience=15)
tuner = kt.BayesianOptimization(model_builder, objective=kt.Objective('val_rmse', direction='min'), max_trials=100,
                                executions_per_trial=2)
tuner.search(x=x_train, y=y_train, epochs=100, batch_size=256, validation_data=(x_test, y_test), callbacks=[es])
tuned_model = tuner.get_best_models(2)
best_models = tuner.get_best_models(2)

# Salvam cele 2 modele optime
for i, model in enumerate(best_models):
    if i != 0:
        model.save(f"models/model_{stock_symbol.lower()}{i}")
    else:
        model.save(f"models/model_{stock_symbol.lower()}")
