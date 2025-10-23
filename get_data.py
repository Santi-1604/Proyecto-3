import yfinance as yf

import warnings
warnings.filterwarnings('ignore')
def get_data(ticker):
    df = yf.download(tickers=ticker, period='15y', interval='1d').dropna()
    return df
df=get_data('ORCL')
print(df)