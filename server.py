import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
from utils import download_stock_df, prepare_data

path = "./models/model_"
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # primim datele de intrare de la client
    input_data = request.get_json()

    # citim ce model trebuie sa rulam
    # daca nu exista, atunci rulam un model default
    stock_symbol = input_data['symbol']
    if os.path.exists(path + stock_symbol):
        model = tf.keras.models.load_model(path + stock_symbol)
    else:
        model = tf.keras.models.load_model(path + "default")

    # preluam datele de input si le scalam
    stock_data_df = download_stock_df(stock_symbol)
    stock_data, _, scaler_input, scaler_output = prepare_data(stock_data_df)

    stock_data = stock_data[-30:]
    stock_data = np.reshape(stock_data, (1, 30, 5))

    # procesăm datele de intrare cu modelul încărcat
    prediction = model.predict(stock_data)
    unscaled_pred = scaler_output.inverse_transform(prediction)
    unscaled_pred = unscaled_pred.tolist()
    result = [val for sublist in unscaled_pred for val in sublist]
    # convertim predicția
    output_data = {'symbol': stock_symbol, 'prediction': result}

    # returnăm rezultatul
    return jsonify(output_data)


# def predict():
#     # primim datele de intrare de la client
#     input_data = request.get_json()
#
#     # citim ce model trebuie sa rulam
#     # daca nu exista, atunci rulam un model default
#     stock_model = input_data['model']
#     if os.path.exists(path+stock_model):
#         model = tf.keras.models.load_model(path+stock_model)
#     else:
#         model = tf.keras.models.load_model(path + "default")
#
#     # preluam datele de input si le scalam
#     inputs = download_stock_df(stock_model)
#     scaler_input = MinMaxScaler()
#     scaler_output = MinMaxScaler()
#     scaled_data = scaler_input.fit_transform(inputs)
#     scaler_output.fit(np.array(inputs)[:, 3].reshape(-1, 1))
#     # transformăm datele de intrare într-un tensor de forma (1, 30, 5)
#     inputs = tf.constant([scaled_data], dtype=tf.float32)
#
#
#
#     # procesăm datele de intrare cu modelul încărcat
#     prediction = model.predict(inputs)
#     unscaled_pred = scaler_output.inverse_transform(prediction)
#     # convertim predicția într-un dicționar cu cheie "output" și valoarea predicției
#     output_data = {'symbol': stock_model, 'prediction': unscaled_pred[:7, 3].tolist()}
#
#     # returnăm rezultatul
#     return jsonify(output_data)


if __name__ == '__main__':
    app.run(port=5000)
