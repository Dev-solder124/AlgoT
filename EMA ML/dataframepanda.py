import pandas as pd 
import pandas_ta as ta
import numpy as np
from scipy.stats import linregress

def linreg_slope(y):
    x = np.arange(len(y))
    slope, _, _, _, _ = linregress(x, y)
    return slope  



df=pd.read_csv(r"D:\AlgoT\sensex_candlestick_data.csv")
# df.columns = df.columns.str.lower()

df=df.round(2)

df['EMA75'] = df['close'].ewm(span=75, adjust=False).mean()

# Price relative to EMA
df['high_ema'] = df['high'] - df['EMA75']                               #1
df['low_ema'] = df['low'] - df['EMA75']                                 #2
df['close_ema'] = df['close'] - df['EMA75']                             #3
print("3 done")
df['slope_close'] = df['close'].rolling(window=30).apply(linreg_slope, raw=True) #14
df['slope_ema'] = df['EMA75'].rolling(window=120).apply(linreg_slope, raw=True)  #15
print("5 done")
# Bollinger Bands (15-min equivalent)
bb=df.ta.bbands(length=300)
df['BBB']=bb['BBB_300_2.0']                                             #4
df['BBP']=bb['BBP_300_2.0']                                             #5
print("7 done")

# MACD (15-min equivalent)
df.ta.macd(fast=180, slow=390, signal=135, append=True)                #6 7 8

# RSI (15-min equivalent)
df.ta.rsi(length=210, append=True)                                     #9

# ATR (15-min equivalent)
df.ta.atr(length=210, append=True)                                     #10
print("10 done")
# ADX (15-min equivalent)
adx=df.ta.adx(length=210)
df['ADX_210']=adx["ADX_210"]                                           #11
df['DMP_DMN']=adx['DMP_210']-adx['DMN_210']                            #12

# CCI (15-min equivalent)
df.ta.cci(length=300, append=True)                                     #13

print(df)
print(df.head())
print(df.info())

# print(bb)
df=df.round(2)

df.to_csv("ML F SENSEX.csv", index=False)
