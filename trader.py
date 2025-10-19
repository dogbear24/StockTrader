# simple_trader.py
import pandas as pd
import alpaca_trade_api as tradeapi
import time
from dotenv import load_dotenv
import os

#IMports api keys
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = 'https://paper-api.alpaca.markets'  # paper trading

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

SYMBOL = 'AAPL'
SHORT_WINDOW = 10
LONG_WINDOW = 50
TRADE_QUANTITY = 1  # shares per trade

def get_historical_data(symbol, limit=100):
    bars = api.get_bars(symbol, tradeapi.TimeFrame.Day, limit=limit).df
    
    if bars.empty:
        print("No data returned from Alpaca. Check your API keys and symbol.")
        return bars
    
    # reset index and normalize column names
    bars = bars.reset_index()
    bars.columns = [c.lower() for c in bars.columns]  # lowercase everything
    
    # If multi-symbol, filter
    if 'symbol' in bars.columns:
        bars = bars[bars['symbol'] == symbol]
    
    return bars

def check_signals(df):
    # SMA logic using lowercase 'close'
    df['sma_short'] = df['close'].rolling(SHORT_WINDOW).mean()
    df['sma_long'] = df['close'].rolling(LONG_WINDOW).mean()
    
    if df['sma_short'].iloc[-2] < df['sma_long'].iloc[-2] and df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1]:
        return 'BUY'
    elif df['sma_short'].iloc[-2] > df['sma_long'].iloc[-2] and df['sma_short'].iloc[-1] < df['sma_long'].iloc[-1]:
        return 'SELL'
    return 'HOLD'

def execute_trade(signal):
    if signal == 'BUY':
        api.submit_order(symbol=SYMBOL, qty=TRADE_QUANTITY, side='buy', type='market', time_in_force='gtc')
        print("Bought 1 share")
    elif signal == 'SELL':
        api.submit_order(symbol=SYMBOL, qty=TRADE_QUANTITY, side='sell', type='market', time_in_force='gtc')
        print("Sold 1 share")
    else:
        print("Holding position")

if __name__ == '__main__':
    while True:
        df = get_historical_data(SYMBOL, limit=100)
        if df.empty:
            time.sleep(60*60)  # wait 1 hour and retry
            continue
        
        print(df.head())
        print(df.columns)
        signal = check_signals(df)
        execute_trade(signal)
        time.sleep(60*60*24)  # run once per day