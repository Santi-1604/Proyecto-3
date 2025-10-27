import yfinance as yf

import warnings
warnings.filterwarnings('ignore')
def get_data(ticker):
    df = yf.download(tickers=ticker, period='15y', interval='1d').dropna()
    df.columns= df.columns.droplevel(1)
    return df

