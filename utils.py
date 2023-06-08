import numpy as np
import pandas as pd
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import time

def download_technical_data(symbol):
    ti = TechIndicators(key='Z3QM56MJPC8SYUT7', output_format='pandas')
    data, _ = ti.get_sma(symbol=symbol, interval='daily', time_period=30)
    return data

def download_economic_data(function, interval = ""):
    # Obține datele JSON de la Alpha Vantage API
    url = ""
    if interval == "":
        url = f'https://www.alphavantage.co/query?function={function}&apikey=Z3QM56MJPC8SYUT7'
    else:
        url = f'https://www.alphavantage.co/query?function={function}&interval={interval}&apikey=Z3QM56MJPC8SYUT7'
    r = requests.get(url)
    data = r.json()
    # Transformă datele JSON într-un DataFrame pandas
    df_gdp = pd.DataFrame(data['data'])
    df_gdp['date'] = pd.to_datetime(df_gdp['date'])
    df_gdp['value'] = pd.to_numeric(df_gdp['value'])
    df_gdp.columns = ['date', function.lower()]
    
    # Setează coloana 'date' ca index și setează frecvența la anual
    df_gdp.set_index('date', inplace=True)
    print(df_gdp)
    return df_gdp

def concat_df(df1, df2, period = 'M'):
    df1['year_month'] = df1.index.to_period(period)
    df2['year_month'] = df2.index.to_period(period)
    # Realizeaza un left join pe coloana 'year_month'
    df_final = pd.merge(df1, df2, left_on='year_month', right_on='year_month', how='left')
    df_final.fillna(method='bfill', inplace=True)
    df_final['date'] = df1.index
    df_final.set_index('date', inplace=True)
    # Elimina coloana 'year_month'
    df_final = df_final.drop(columns=['year_month'])
    return df_final

# download open, high, low, close, volume data from api, outputsize = full (all data range)
def download_stock_df(stock_name) -> DataFrame:
    ts = TimeSeries(key='Z3QM56MJPC8SYUT7', output_format='pandas', indexing_type='date')
    stock_data_df, _ = ts.get_daily_adjusted(stock_name, outputsize='full')
    stock_data_df.drop(['5. adjusted close', '7. dividend amount', '8. split coefficient'], axis=1, inplace=True)
    stock_data_df.columns = ["open", "high", "low", "close", "volume"]
    stock_data_df = concat_df(stock_data_df, download_economic_data("REAL_GDP", "quarterly"))
    stock_data_df = concat_df(stock_data_df, download_economic_data("CPI", "monthly"))
    stock_data_df = concat_df(stock_data_df, download_economic_data("INFLATION"))
    stock_data_df = concat_df(stock_data_df, download_economic_data("UNEMPLOYMENT"))
    print("Sleeping for 70 seconds for API call limit")
    time.sleep(70)
    stock_data_df = concat_df(stock_data_df, download_economic_data("FEDERAL_FUNDS_RATE", "monthly"))
    stock_data_df = concat_df(stock_data_df, download_technical_data(stock_name), "D")
    stock_data_df = concat_df(stock_data_df, download_technical_data(stock_name), "D")
    stock_data_df.dropna(inplace=True)
    return stock_data_df.iloc[::-1]


# Filtreaza si scaleaza datele
def prepare_data(df: DataFrame):
    # Ce features vrem sa luam din dataset
    features = ['open', 'high', 'low', 'close', 'volume'] #, 'real_gdp', 'cpi', 'inflation', 'unemployment']

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
