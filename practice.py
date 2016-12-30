from yahoo_finance import Share
from pprint import pprint
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import pandas_datareader.data as web
from datetime import datetime
import os
import matplotlib.pyplot as plt

def symbol_to_path (symbol, base_dir = 'data'):
    return  os.path.join ( base_dir, '{}.csv'.format(symbol))

def get_data (symbol, dates):
    df = pd.DataFrame(index=dates)
    df_temp = pd.read_csv (symbol_to_path(symbol),index_col="Date", parse_dates=True, na_values=['nan'])
    df = df.join(df_temp)
    df =df.dropna()
    return df

def get_close_data (symbols, dates):
    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv (symbol_to_path(symbol),index_col="Date", parse_dates=True, usecols=['Date','Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        df =df.dropna()
    return df

def plot_data(data, title = 'Stock Prices', value = None):
    if value is None:
        value = data.columns.values.tolist()
        print type(value)
        value.remove('Volume')
    ax = data[value].plot(title=title,fontsize = 12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Prices")
    plt.show()


def run():
    # Run function which is called from main.
    # print symbol_to_path('AMZN')
    # df = get_data("AMZN", pd.date_range('2009-1-1','2009-3-1'))
    # print df
    # plot_data(df, value='Volume')
    df = pd.DataFrame(data=None,columns=['First','Second'])
    df['0'] = [1,2]
    print df


if __name__ == '__main__':
    run()



