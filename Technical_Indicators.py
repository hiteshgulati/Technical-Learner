__author__ = "Hitesh Gulati"

import pandas as pd
import numpy as np
import math
import random
import os
import matplotlib.pyplot as plt
from pprint import pprint
from datetime import datetime
from datetime import timedelta
import statsmodels.api as sm
import talib

Adj_Close = "Adj Close"
High = "High"
Low = "Low"
Close = "Close"
Open = "Open"
Volume = "Volume"
Date = "Date"
Uptrend = "Uptrend"
Downtrend = "Downtrend"
Notrend = "Notrend"
Buy = "Buy"
Sell = "Sell"
Hold = "Hold"
Strong_Volume = "Strong Volume"
Weak_Volume = "Weak Volume"
Weak_Trend = "Weak Trend"
Strong_Trend = "Strong Trend"
list_of_stocks_filename = "summary_file.csv"
Poor = "Poor"
Bad ="Bad"
Ok = "Ok"
Good = "Good"
prediction_days = 7
transaction_cost = 0
risk_appetite = .5
Decision_Hold = 0
Decision_Buy = 1
Decision_Sell = -1
def symbol_to_path (symbol, data_dir = 'data'):
    return  os.path.join ( data_dir, '{}.csv'.format(symbol))

# Get data of single stock by passing symbol and dates from data directory
def get_data (symbol, dates):
    df = pd.DataFrame(index=dates)
    df_temp = pd.read_csv (symbol_to_path(symbol),index_col=Date, parse_dates=True, na_values=['nan'])
    df = df.join(df_temp)
    df =df.dropna()
    return df
# Get Adj Close data of list of stocks and combin as one data renaming Adj CLose to Ticker value
def get_close_data (symbols, dates):
    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv (symbol_to_path(symbol),index_col=Date, parse_dates=True, usecols=[Date,Adj_Close], na_values=['nan'])
        df_temp = df_temp.rename(columns={Date: symbol})
        df = df.join(df_temp)
        df =df.dropna()
    return df
# Plot O,H,L,C,AC values or individual values if mentioned
def plot_data(data, title = 'Stock Prices', measure = None):
    if measure is None:
        measure = data.columns.values.tolist()
        measure.remove(Volume)
    ax = data[measure].plot(title=title,fontsize = 12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Prices")
    plt.show()
# Get list of measures to be used as per given standards
def get_measures (data, measures):
    if measures is "All":
        measures = data.columns.values.tolist()
    elif isinstance(measures,basestring):
        measures = [measures]
    elif measures is None:
        measures = [Adj_Close]
    return measures
# alter column names accoring to measure calculated
def alter_column_names (label, data, measures, window):
    new_column_names = {}
    for measure in measures:
        new_column_names[measure] = str(label) + str(window) + "_" + measure
    data = data.rename(columns=new_column_names)
    return data
# return indicator seprately or added to set as requirment
def add_or_not (data, indicator, add_to_data, label="New Value"):
    if add_to_data is False:
        return indicator
    elif add_to_data is True:
        return add_indicator(data,indicator,label)

# Join data set with indicator
def add_indicator (data, indicator, label = "New Value"):
    if type(indicator) is pd.core.frame.DataFrame:
        return data.join(indicator)
    else:
        data[label] = indicator
# Get Moving Average of mentioned size default 14 and add to data if True
def MA (data, window =14, measures = None, add_to_data = False ):
    measures = get_measures(data, measures)
    indicator = data[measures].rolling(window).mean()
    indicator = alter_column_names("MA", indicator,measures,window )
    return add_or_not(data,indicator,add_to_data,label=measures[0])

# Get Chaikin Money Flow
def CMF (data, window=14, measures = None, add_to_data = False):
    label = "CMF" + str (window)
    money_flow_multiplier = ((data[Close]-data[Low])-(data[High]-data[Close]))/(data[High]-data[Low])
    money_flow_volume = money_flow_multiplier * data[Volume]
    new_data = data.copy()
    new_data["Money Flow Volume"] = money_flow_volume
    Chaikin_Money_Flow = (new_data["Money Flow Volume"].rolling(window).sum())/(new_data[Volume].rolling(window).sum())
    return add_or_not(data,Chaikin_Money_Flow,add_to_data,label=label)

# Money Flow Index
def MFI(data, window=14, measures = None, add_to_data = False):
    label = "MFI" + str (window)
    typical_price = (data[High] + data [Low] + data [Close])/3
    raw_money_flow = typical_price * data[Volume]
    new_data = data.copy()
    new_data["Typical Price"] = typical_price
    new_data ["Raw Money Flow"] = raw_money_flow
    new_data ["PosMF"] = raw_money_flow
    new_data ["NegMF"] = raw_money_flow * 1
    #print new_data
    i=1
    date = new_data.index [i]

    while i < len(new_data):
        previous_date = new_data.index [i-1]
        if new_data.ix[date,["Typical Price"]][0] >= new_data.ix[previous_date,["Typical Price"]][0]:
            new_data.ix[date,["NegMF"]]  = 0
        else:
            new_data.ix[date, ["PosMF"]] = 0
        if i < len(new_data)-1:
            date = new_data.index[i+1]
        i += 1
    money_flow_ratio = new_data["PosMF"].rolling(window).sum()/new_data["NegMF"].rolling(window).sum()
    new_data["Money Flow Ratio"] =  money_flow_ratio
    money_flow_index = 100 - (100/(1+money_flow_ratio))
    new_data["Money Flow Index"]= money_flow_index
    return add_or_not(data,money_flow_index,add_to_data,label= label)
# Accumulaion/Distribution Line
def ADL (data, window=1, measures = None, add_to_data = False):
    label = "ADL" + str(window)
    money_flow_multiplier = ((data[Close] - data[Low]) - (data[High] - data[Close])) / (data[High] - data[Low])
    money_flow_volume = money_flow_multiplier * data[Volume]
    adl = money_flow_volume.copy()
    i = window
    while i < len(adl):
        adl [i] = money_flow_volume[i] + adl[i-window]
        i += 1
    #print adl
    return add_or_not(data,adl,add_to_data,label = label)

#Average True Range
def ATR (data, window=14, measures = None, add_to_data = False):
    label = "ATR" + str(window)
    HminusL = data[High] - data [Low]
    HminusCp = abs(data[High] - data[Close].shift(1))
    LminusCp = abs(data[Low]-data[Close].shift(1))
    true_range = HminusL
    i = 1
    while i < len(true_range):
        true_range[i] = max (HminusL[i],HminusCp[i],LminusCp[i])
        i += 1
    i = window
    adr = true_range.rolling(window).mean()
    while i< len(adr):
        adr[i] = (adr[i-1]*(window-1) + true_range[i])/window
        i += 1
    return add_or_not(data,adr,add_to_data,label = label)

def absolute_sum (data):
    absolute_data = np.absolute(data)
    return absolute_data.sum()

#Hodrick-Prescott Filter. Returns the trend Uptrend, Downtrend or Notrend
def HP_trend_agent (data, window =14, measures = Adj_Close):
    label = "HPtrend_agent" + str(window)
    lamb = 5000
    cycle_value, trend_value = sm.tsa.filters.hpfilter(data[Adj_Close],lamb)
    trend_value_firstdiff = trend_value - trend_value.shift(1)
    trend_string = [Downtrend for x in range(len(trend_value))]
    trend = np.array(trend_string)
    trend_variation = trend_value_firstdiff.rolling(window).std() / trend_value_firstdiff.rolling(window).mean()
    trend_sum = trend_value_firstdiff.rolling(window).sum()
    trend_abs_sum = trend_value_firstdiff.rolling(window).apply(absolute_sum)
    for i in range(window,len(trend)):
        if trend_variation[i] <= .05:
            if  trend_sum[i] == trend_abs_sum[i]:
                trend[i]= Uptrend
            elif trend_sum[i] == -trend_abs_sum[i]:
                trend[i]= Downtrend
        else:
            trend[i] = Notrend
    trend_uptrend = trend.copy()
    trend_downtrend = trend.copy()
    trend_notrend = trend.copy()
    for i in range (0,len(trend)):
        if trend[i]== Uptrend:
            trend_uptrend[i] = 1
            trend_downtrend[i] = 0
            trend_notrend[i] = 0
        elif trend[i] == Downtrend:
            trend_uptrend[i] = 0
            trend_downtrend[i] = 1
            trend_notrend[i] = 0
        elif trend[i] == Notrend:
            trend_uptrend[i] = 0
            trend_downtrend[i] = 0
            trend_notrend[i] = 1
        else:
            trend_uptrend[i] = 0
            trend_downtrend[i] = 0
            trend_notrend[i] = 0
    Hp_signal = data.copy()
    Hp_signal['HP_Notrend'] = trend_notrend
    Hp_signal['HP_Uptrend'] = trend_uptrend
    Hp_signal['HP_Downtrend'] = trend_downtrend
    return Hp_signal

#Stochastic Agent. Returns Buy, Sell or Hold.
def stochastic_agent (data, window =5, measures = Adj_Close):
    label = "stochastic_agent" + str(window)
    slowk, slowd = talib.STOCH (data[High].values, data[Low].values, data[Close].values,fastk_period=window)
    stochastic_string = [Hold for x in range(len(data))]
    stochastic_signal = np.array(stochastic_string)
    i = window
    while i < len(stochastic_signal):
        if slowk[i-1] < slowd[i-1] and slowd[i-1] < 20:
            if slowd[i] < slowk[i] and slowk[i] < 20:
                stochastic_signal[i] = Buy
        elif slowk[i-1] > slowd[i-1] and slowd[i-1] > 80:
            if slowd[i] > slowk[i] and slowk[i] > 80:
                stochastic_signal[i]= Sell
        i += 1
    trend = stochastic_signal
    trend_Buy = trend.copy()
    trend_Sell = trend.copy()
    trend_Hold = trend.copy()
    for i in range(0, len(trend)):
        if trend[i] == Buy:
            trend_Buy[i] = 1
            trend_Sell[i] = 0
            trend_Hold[i] = 0
        elif trend[i] == Sell:
            trend_Buy[i] = 0
            trend_Sell[i] = 1
            trend_Hold[i] = 0
        elif trend[i] == Hold:
            trend_Buy[i] = 0
            trend_Sell[i] = 0
            trend_Hold[i] = 1
        else:
            trend_Buy[i] = 0
            trend_Sell[i] = 0
            trend_Hold[i] = 0
    Stochastic_signal = data.copy()
    Stochastic_signal['Stochastic_Buy'] = trend_Buy
    Stochastic_signal['Stochastic_Sell'] = trend_Sell
    Stochastic_signal['Stochastic_Hold'] = trend_Hold
    return Stochastic_signal
    # return add_or_not(data,stochastic_agent,add_to_data=add_to_data, label =label)

#Moving Average Crossover Agent. Returns Buy, Sell or Hold.
def MAC_agent (data, window =14, measures = Adj_Close):
    label = "MAC_agent" + str(window)
    short_window = window
    long_window = ((window/7)/2)*35
    short_moving_average = data[measures].rolling(short_window).mean()
    long_moving_average = data[measures].rolling(long_window).mean()
    MAC_string = [Hold for x in range(len(data))]
    MAC_signal = np.array(MAC_string)
    for i in range(long_window,len(MAC_signal)):
        if short_moving_average[i-1] < long_moving_average[i-1]:
            if short_moving_average[i] > long_moving_average[i]:
                MAC_signal[i] = Buy
        elif short_moving_average[i-1] > long_moving_average[i-1]:
            if short_moving_average[i] < long_moving_average [i]:
                MAC_signal[i] = Sell
    trend = MAC_signal
    trend_Buy = trend.copy()
    trend_Sell = trend.copy()
    trend_Hold = trend.copy()
    for i in range(0, len(trend)):
        if trend[i] == Buy:
            trend_Buy[i] = 1
            trend_Sell[i] = 0
            trend_Hold[i] = 0
        elif trend[i] == Sell:
            trend_Buy[i] = 0
            trend_Sell[i] = 1
            trend_Hold[i] = 0
        elif trend[i] == Hold:
            trend_Buy[i] = 0
            trend_Sell[i] = 0
            trend_Hold[i] = 1
        else:
            trend_Buy[i] = 0
            trend_Sell[i] = 0
            trend_Hold[i] = 0
    MAC_signal = data.copy()
    MAC_signal['MAC_Buy'] = trend_Buy
    MAC_signal['MAC_Sell'] = trend_Sell
    MAC_signal['MAC_Hold'] = trend_Hold
    return MAC_signal

#Volume Agent. Returns Strong Volume or Weak Volume
def volume_agent (data, window =14, measures = Volume):
    label = "volume_agent" + str(window)
    long_window = window
    short_window = (long_window /7)+1
    short_moving_average = data[measures].rolling(short_window).mean()
    long_moving_average = data[measures].rolling(long_window).mean()
    volume_string = [Strong_Volume for x in range(len(data))]
    volume_signal = np.array(volume_string)
    i = long_window
    while i < len(volume_signal):
        if long_moving_average[i] > short_moving_average[i]:
            volume_signal[i] = Weak_Volume
        i += 1
    trend = volume_signal
    trend_weak = trend.copy()
    trend_strong = trend.copy()
    for i in range(0, len(trend)):
        if trend[i] == Weak_Volume:
            trend_weak[i] = 1
            trend_strong[i] = 0
        elif trend[i] == Strong_Volume:
            trend_weak[i] = 0
            trend_strong[i] = 1
        else:
            trend_weak[i] = 0
            trend_strong[i] = 0
    return_signal = data.copy()
    return_signal['Volume_Weak'] = trend_weak
    return_signal['Volume_Strong'] = trend_strong
    return return_signal


# Average Directional Index Agent, Returns points depending on the strength of the trend. Higher strength higher points.
def ADX_agent (data, window =14, measures = Adj_Close):
    label = "ADX_agent" + str(window)
    ADX = talib.ADX(data[High].values,data[Low].values,data[Low].values,timeperiod=window)
    trend_string = [Strong_Trend for x in range(len(data))]
    trend_signal = np.array(trend_string)
    for i in range(window, len(trend_signal)):
        if ADX[i] <= 25:
            trend_signal[i] = Weak_Trend
        else:
            trend_signal[i] = Strong_Trend
    trend = trend_signal
    trend_weak = trend.copy()
    trend_strong = trend.copy()
    for i in range(0, len(trend)):
        if trend[i] == Weak_Trend:
            trend_weak[i] = 1
            trend_strong[i] = 0
        elif trend[i] == Strong_Trend:
            trend_weak[i] = 0
            trend_strong[i] = 1
        else:
            trend_weak[i] = 0
            trend_strong[i] = 0
    return_signal = data.copy()
    return_signal['Trend_Weak'] = trend_weak
    return_signal['Trend_Strong'] = trend_strong
    return return_signal

# Candlestick agent. Returns Buy, Sell or Hold.
def Candlestick_agent (data, window =14, measures = Adj_Close):
    label = "Candlestick_agent"
    indicators = {}
    indicators['sum'] = 0
    indicators['hammer']= talib.CDLHAMMER(data[Open].values,data[High].values,data[Low].values,data[Close].values)
    indicators['engulfing'] = talib.CDLENGULFING(data[Open].values,data[High].values,data[Low].values,data[Close].values)
    indicators['hangingman'] = talib.CDLHANGINGMAN(data[Open].values,data[High].values,data[Low].values,data[Close].values)
    indicators['threewhitesoldiers'] = talib.CDL3WHITESOLDIERS(data[Open].values,data[High].values,data[Low].values,data[Close].values)
    indicators['threeblackcrows'] = talib.CDL3BLACKCROWS(data[Open].values,data[High].values,data[Low].values,data[Close].values)
    indicators['threeinside'] = talib.CDL3INSIDE(data[Open].values,data[High].values,data[Low].values,data[Close].values)
    indicators['eveningstar'] = talib.CDLEVENINGSTAR(data[Open].values,data[High].values,data[Low].values,data[Close].values)
    indicators['morningstar'] = talib.CDLMORNINGSTAR(data[Open].values,data[High].values,data[Low].values,data[Close].values)
    indicators['threeoutside'] = talib.CDL3OUTSIDE(data[Open].values,data[High].values,data[Low].values,data[Close].values)

    for value in indicators:
        indicators['sum'] += indicators[value]
    CDL_string = [Hold for x in range(len(data))]
    CDL_signal = np.array(CDL_string)
    for i in range(0,len(CDL_signal)):
        if indicators['sum'][i] > 0:
            CDL_signal[i] = Buy
        elif indicators['sum'][i] < 0:
            CDL_signal[i] = Sell
    trend = CDL_signal
    trend_Buy = trend.copy()
    trend_Sell = trend.copy()
    trend_Hold = trend.copy()
    for i in range(0, len(trend)):
        if trend[i] == Buy:
            trend_Buy[i] = 1
            trend_Sell[i] = 0
            trend_Hold[i] = 0
        elif trend[i] == Sell:
            trend_Buy[i] = 0
            trend_Sell[i] = 1
            trend_Hold[i] = 0
        elif trend[i] == Hold:
            trend_Buy[i] = 0
            trend_Sell[i] = 0
            trend_Hold[i] = 1
        else:
            trend_Buy[i] = 0
            trend_Sell[i] = 0
            trend_Hold[i] = 0
    return_signal = data.copy()
    return_signal['CDL_Buy'] = trend_Buy
    return_signal['CDL_Sell'] = trend_Sell
    return_signal['CDL_Hold'] = trend_Hold
    return return_signal

# Returntorisk agent. Return Bad, Ok, Good, Excellent, based on returntorisk ratio.
def RRR_agent (data, window =7, measures = Adj_Close):
    label = "returntorisk_agent" + str(window)
    return_expost = (data[Adj_Close]-data[Adj_Close].shift(window))/data[Adj_Close].shift(window)
    risk_expost = return_expost.rolling(window).std()
    returntorisk_expost = return_expost/risk_expost
    returntorisk_string = [Good for x in range(len(data))]
    returntorisk_signal = np.array(returntorisk_string)
    for i in range (window, len(returntorisk_signal)):
        if abs(returntorisk_expost[i]) < risk_appetite:
            returntorisk_signal[i] = Bad
        elif abs(returntorisk_expost[i]) < risk_appetite*4:
            returntorisk_signal[i] = Ok
        elif abs(returntorisk_expost[i]) < risk_appetite*4:
            returntorisk_signal[i] = Good
    trend = returntorisk_signal
    trend_Bad = trend.copy()
    trend_Ok = trend.copy()
    trend_Good = trend.copy()
    for i in range(0, len(trend)):
        if trend[i] == Bad:
            trend_Bad[i] = 1
            trend_Ok[i] = 0
            trend_Good[i] = 0
        elif trend[i] == Ok:
            trend_Bad[i] = 0
            trend_Ok[i] = 1
            trend_Good[i] = 0
        elif trend[i] == Good:
            trend_Bad[i] = 0
            trend_Ok[i] = 0
            trend_Good[i] = 1
        else:
            trend_Bad[i] = 0
            trend_Ok[i] = 0
            trend_Good[i] = 0
    return_signal = data.copy()
    return_signal['RRR_Bad'] = trend_Bad
    return_signal['RRR_Ok'] = trend_Ok
    return_signal['RRR_Good'] = trend_Good
    return return_signal

# The correct decision. Based on exAnte calculations of return and risk.
def correct_decision (data, window =7, measures = Adj_Close):
    window = prediction_days
    label = "new_decision" + str(window)
    return_exante = (data[Adj_Close] - data[Adj_Close].shift(window - 1)) / data[Adj_Close].shift(window - 1)
    risk_exante = return_exante.rolling(window).std()
    returntorisk_exante = return_exante / risk_exante
    decision_string = [Hold for x in range(len(data))]
    decision_signal = np.array(decision_string)
    for i in range(window, len(decision_signal)-window):
        if abs(returntorisk_exante[i]) >= risk_appetite:
            if return_exante[i] < -transaction_cost :
                decision_signal[i-window] = Decision_Sell
            if return_exante[i] > transaction_cost:
                decision_signal[i-window] = Decision_Buy
        else:
            decision_signal[i-window] = Decision_Hold
    # new_data = pd.DataFrame(data[Adj_Close])
    # new_data['return'] = return_exante
    # new_data['risk'] = risk_exante
    # new_data['sharpe'] = returntorisk_exante
    # new_data[label] = decision_signal
    # print new_data
    return_signal = data.copy()
    return_signal['Decision'] = decision_signal
    return  return_signal


def make_indicator_files ():
    summary_file_name = "data/" + list_of_stocks_filename
    ticker_file = pd.read_csv(summary_file_name)
    for i in range(0,len(ticker_file)):
        company_name = ticker_file['Company_Name'][i]
        company_ticker = ticker_file['Ticker'][i]
        symbol = company_ticker
        df = get_data(symbol, pd.date_range('2006-1-1', '2014-1-1'))
        print "Data Read for: ", company_name , " - ", company_ticker
        df = HP_trend_agent(df)
        df = MAC_agent(df)
        df = stochastic_agent(df)
        df = volume_agent(df)
        df = ADX_agent(df)
        df = Candlestick_agent(df)
        df = RRR_agent(df)
        df = correct_decision(df)
        print "Indicators made for: ", company_name, " - ", company_ticker
        columns = df.columns.values.tolist()
        # columns.insert(0, 'Date')
        df.to_csv(symbol_to_path("i_"+symbol),index_label="Date")
        print "File made: i_",symbol
    print "Done!"

def generate_test_train ():
    summary_file_name = "data/" + list_of_stocks_filename
    ticker_file = pd.read_csv(summary_file_name)
    company_name = ticker_file['Company_Name'][0]
    company_ticker = "i_" + ticker_file['Ticker'][0]
    symbol = company_ticker
    df_training = get_data(symbol, pd.date_range('2006-3-1', '2012-3-1'))
    df_testing = get_data(symbol, pd.date_range('2012-3-1', '2013-12-1'))
    print "Data Read for: ", company_name , " - ", company_ticker
    training_data = df_training.copy()
    testing_data = df_testing.copy()
    training_size = len(training_data)
    testing_size = len(testing_data)
    print "training and testing data made"
    for i in range(1,len(ticker_file)):
        company_name = ticker_file['Company_Name'][i]
        company_ticker = "i_" + ticker_file['Ticker'][i]
        symbol = company_ticker
        df_training = get_data(symbol, pd.date_range('2006-3-1', '2012-3-1'))
        df_testing = get_data(symbol, pd.date_range('2012-3-1', '2013-12-1'))
        print "Data Read for: ", company_name , " - ", company_ticker, "Training size: ", len(df_training), "Testing size: ", len(df_testing)
        training_data = training_data.append(df_training,ignore_index=True)
        testing_data = testing_data.append(df_testing,ignore_index=True)
        print "training and testing data made"
        training_size += len(df_training)
        testing_size += len(df_testing)
    columns = training_data.columns.values.tolist()
    columns = columns [6:]
    print columns
    training_data.to_csv(symbol_to_path("training_data"),columns=columns)
    testing_data.to_csv(symbol_to_path("testing_data"),columns=columns)
    print "training size: ", training_size
    print "testing_size: ", testing_size
    print "training file size: ", len(training_data)
    print "testing file size: ", len(testing_data)
    print " Done!!!"

    return None

def run():
    # Run function which is called from main.
    symbol = 'BBW'
    print symbol_to_path(symbol)
    df = get_data(symbol, pd.date_range('2012-1-1', '2012-04-1'))
    print "Data Read"
    #print df
    #df =  MA(df, window =14, add_to_data=True)
    #plot_data(df, measure=[Adj_Close,'MA14_Adj Close'])
    #CMF(df,add_to_data=True,window=20)
    #print df
    # plot_data(df, measure="CMF14")
    #MFI (df, add_to_data= True)
    #plot_data(df, measure= ["Adj Close","MFI14"])
    #ADL(df,add_to_data=True)
    #print df
    # ATR(df,add_to_data=True)
    # plot_data(df,measure="ATR14")
    # summary_file_name = "data/summary_file.csv"
    # ticker_file = pd.read_csv(summary_file_name)
    # for i in range(0,len(ticker_file)):
    #     company_name = ticker_file['Company_Name'][i]
    #     company_ticker = ticker_file['Ticker'][i]
    #     symbol = company_ticker
    #     df = get_data(symbol, pd.date_range('2006-1-1', '2012-11-1'))
    #     print "Data Read for: ", company_name , " - ", company_ticker
    #     trend = trend_agent(df)
    # print "Done!"
    # correct_decision(df, add_to_data=True)
    # print df

    #make_indicator_files()
    generate_test_train()

if __name__ == '__main__':
    run()
