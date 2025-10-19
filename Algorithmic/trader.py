# simple_trader.py
import pandas as pd

#Have separate venv for alpaca due to verion comflict on webcokets with yfinance
#import alpaca_trade_api as tradeapi
import yfinance as yf
import time
from dotenv import load_dotenv
import os
import numpy as np

#IMports api keys
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = 'https://paper-api.alpaca.markets'  # paper trading

SYMBOL = 'INTC'
START = "2010-01-01"
END = "2025-01-01"
SHORT_WINDOW = 10
LONG_WINDOW = 50
#TRADE_QUANTITY = 100  # shares per trade


####################################################################################################
# Load historical data CSV or fetched data
df = yf.download(SYMBOL, start=START, end=END)
df.to_csv("BackTesting_Data/AAPL.csv")

#Clean Data
# Ensure numeric columns only
df = df[['Open','High','Low','Close','Volume']].copy()
df = df.dropna()
df.columns = [c[0].lower() for c in df.columns]  # lowercase


df['sma_short'] = df['close'].rolling(SHORT_WINDOW).mean()
df['sma_long'] = df['close'].rolling(LONG_WINDOW).mean()

cash = 10000
shares = 0
df['portfolio_value'] = 0.0
df.loc[df.index[:LONG_WINDOW], 'portfolio_value'] = cash

for i in range(LONG_WINDOW, len(df)):
    # Buy signal: short SMA crosses above long SMA
    if df['sma_short'].iloc[i-1] < df['sma_long'].iloc[i-1] and df['sma_short'].iloc[i] > df['sma_long'].iloc[i]:
        # Buy as many shares as possible
        shares_to_buy = int(cash / df['close'].iloc[i])
        if shares_to_buy > 0:
            cash -= shares_to_buy * df['close'].iloc[i]
            shares += shares_to_buy

    # Sell signal: short SMA crosses below long SMA
    elif df['sma_short'].iloc[i-1] > df['sma_long'].iloc[i-1] and df['sma_short'].iloc[i] < df['sma_long'].iloc[i]:
        # Sell all shares
        if shares > 0:
            cash += shares * df['close'].iloc[i]
            shares = 0
            
    # Record portfolio value for this day
    df.loc[df.index[i], 'portfolio_value'] = cash + shares * df['close'].iloc[i]

            
# Calculate daily returns
df['daily_return'] = df['portfolio_value'].pct_change()
#Measure Volatility
volatility = df['daily_return'].std() * (252 ** 0.5)  # annualized
print(f"Annualized volatility: {volatility:.2%}")

#Calc max drawdown
df['cumulative_max'] = df['portfolio_value'].cummax()
df['drawdown'] = (df['portfolio_value'] - df['cumulative_max']) / df['cumulative_max']
max_drawdown = df['drawdown'].min()
print(f"Maximum drawdown: {max_drawdown:.2%}")

# Annualized Sharpe ratio
avg_daily_return = df['daily_return'].mean()
daily_vol = df['daily_return'].std()
sharpe_ratio = (avg_daily_return / daily_vol) * np.sqrt(252)

print(f"Annualized Sharpe ratio: {sharpe_ratio:.2f}")

final_worth = cash + shares * df['close'].iloc[-1]
print(f"Final net worth: ${final_worth:.2f}")

##################################################################################################################

#api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# def get_historical_data(symbol, limit=100):
#     bars = api.get_bars(symbol, tradeapi.TimeFrame.Day, limit=limit).df
    
#     if bars.empty:
#         print("No data returned from Alpaca. Check your API keys and symbol.")
#         return bars
    
#     # reset index and normalize column names
#     bars = bars.reset_index()
#     bars.columns = [c.lower() for c in bars.columns]  # lowercase everything
    
#     # If multi-symbol, filter
#     if 'symbol' in bars.columns:
#         bars = bars[bars['symbol'] == symbol]
    
#     return bars

# def check_signals(df):
#     # SMA logic using lowercase 'Close'
#     df['sma_short'] = df['Close'].rolling(SHORT_WINDOW).mean()
#     df['sma_long'] = df['Close'].rolling(LONG_WINDOW).mean()
    
#     if df['sma_short'].iloc[-2] < df['sma_long'].iloc[-2] and df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1]:
#         return 'BUY'
#     elif df['sma_short'].iloc[-2] > df['sma_long'].iloc[-2] and df['sma_short'].iloc[-1] < df['sma_long'].iloc[-1]:
#         return 'SELL'
#     return 'HOLD'

# def execute_trade(signal):
#     if signal == 'BUY':
#         api.submit_order(symbol=SYMBOL, qty=TRADE_QUANTITY, side='buy', type='market', time_in_force='gtc')
#         print("Bought 1 share")
#     elif signal == 'SELL':
#         api.submit_order(symbol=SYMBOL, qty=TRADE_QUANTITY, side='sell', type='market', time_in_force='gtc')
#         print("Sold 1 share")
#     else:
#         print("Holding position")

# if __name__ == '__main__':
#     while True:
#         df = get_historical_data(SYMBOL, limit=100)
#         if df.empty:
#             time.sleep(60*60)  # wait 1 hour and retry
#             continue
        
#         print(df.head())
#         print(df.columns)
#         signal = check_signals(df)
#         execute_trade(signal)
#         time.sleep(60*60*24)  # run once per day