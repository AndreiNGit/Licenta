import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from utils import download_stock_df, prepare_data, partition_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = download_stock_df("AAPL")
print(data)
index_Close = data.columns.get_loc("close")
np_scaled, np_scaled_c, scaler_train, scaler_pred = prepare_data(data)
model: tf.keras.Sequential = tf.keras.models.load_model('models/model_default')

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
# joblib.dump(scaler_pred, "scaler.gz")

# Mean Absolute Error (MAE)
MAE = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
print(f'Mean Absolute Error (MAE): {np.round(MAE, 2)}')

# Mean Squared Error (MSE)
MSE = mean_squared_error(y_test_unscaled, y_pred_unscaled)
print(f'Mean Squared Error (MSE): {np.round(MSE, 2)}')

# Root Mean Squared Error (RMSE)
RMSE = mean_squared_error(y_test_unscaled, y_pred_unscaled, squared=False)
print(f'Root Mean Squared Error (RMSE): {np.round(RMSE, 2)}')

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean(np.abs((y_test_unscaled - y_pred_unscaled) / y_test_unscaled)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

# R Squared (R^2)
R2 = r2_score(y_test_unscaled, y_pred_unscaled)
print(f'R Squared (R^2): {np.round(R2, 2)}')

# Predictive direction accuracy
direction_accuracy = np.mean((np.sign(y_test_unscaled[1:] - y_test_unscaled[:-1]) == np.sign(y_pred_unscaled[1:] - y_pred_unscaled[:-1])).astype(int)) * 100
print(f'Predictive Direction Accuracy: {np.round(direction_accuracy, 2)} %')



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
