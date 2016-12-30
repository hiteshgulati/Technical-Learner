""" Utility Functions"""

import random
import math
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from yahoo_finance import Share
from pprint import pprint

def symbol_to_path (symbol, base_dir = 'data'):
    return  os.path.join ( base_dir, '{}.csv'.format(symbol))
def get_data (symbols, dates):
    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv (symbol_to_path(symbol),index_col="Date", parse_dates=True, usecols=['Date','Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        df =df.dropna()
    return df
def fill_missing_values (data):
    data.fillna(method='ffill')
    data.fillna(method= 'backfill')
    return None
def plot_data(data, title = 'Stock Prices'):
    ax = data.plot(title=title,fontsize = 12)
    ax.set_xlabe("Date")
    ax.set_ylabel("Prices")
    plt.show()

def get_rolling_mean (values, window=20):
    return values.rolling(window).mean()

def get_rolling_std (values, window=20):
    return values.rolling(window).std()

def get_bollinger_band (values, window = 20, size = 2):
    upper_band = get_rolling_mean(values,window) + get_rolling_std(values,window) * 2
    lower_band = get_rolling_mean(values, window) + get_rolling_std(values, window) * 2
    return upper_band , lower_band

def get_daily_returns (data):
    data_daily_return = (data/data.shift(1))-1
    return data_daily_return



def run():
    # Run function which is called from main.
    print symbol_to_path('AMZN')
    df = get_data(["AMZN"], pd.date_range('2009-1-1','2009-3-1'))
    print df


if __name__ == '__main__':
    run()
