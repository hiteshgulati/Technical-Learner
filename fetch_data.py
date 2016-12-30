from yahoo_finance import Share
from pprint import pprint
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import pandas_datareader.data as web
from datetime import datetime

ticker_file_name = "data/nasdaq_retail.csv"
summary_file_name = "data/summary_file.csv"
ticker_file = pd.read_csv(ticker_file_name)
ticker_summary_list = []

start = datetime(2006,1,1)
end = datetime(2014,1,1)
print ticker_file['Company Name'][0]
for i in range(0,len(ticker_file)):
    company_name = ticker_file['Company Name'][i]
    company_ticker = ticker_file['Ticker'][i]
    print company_name, company_ticker
    historical_data = pd.DataFrame()
    try:
        historical_data = web.DataReader(company_ticker,data_source='yahoo',start=start,end=end)
        output_file_name = "data/" + company_ticker + ".csv"
        historical_data.to_csv (output_file_name, sep=',')
        ticker_summary = (company_name, company_ticker, len(historical_data))
        ticker_summary_list.append(ticker_summary)
        print ticker_summary
    except pdr._utils.RemoteDataError:
        print "Data Read Error"

ticker_summary_df = pd.DataFrame(ticker_summary_list, columns=['Company_Name', 'Ticker','Number of Entries'])
ticker_summary_df.to_csv(summary_file_name)
