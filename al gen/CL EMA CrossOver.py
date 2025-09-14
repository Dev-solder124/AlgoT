import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dateutil.parser import parse

# Initialize variables
net = 0
Ptrade = 0
Ltrade = 0
n = 375
day = 0

start_day = 1 + (day * n)

# Strategy parameters
initial_target_multiplier = 30.0  # Initial Risk:Reward = 1:3.0
initial_stoploss_multiplier = 1.0
trailing_trigger_ratio = 1.5  # Start trailing when profit >= 1.5x initial risk
trailing_step = 5  # Trail SL by 5 points when price moves favorably

net_data = []
net_data_time = []

Ppoints = 0
Lpoints = 0

print_process = 1

# Load data
df = pd.read_csv(r"D:\AlgoT\NIFTY 50 COPY.csv", usecols=range(6))

datetime_strings = df['date']

# Parse datetime
datetime_objects = []
for dt in datetime_strings:
    try:
        obj = datetime.strptime(dt, '%d-%m-%Y %H:%M')
    except ValueError:
        obj = parse(dt)
    datetime_objects.append(obj)

open_values = df['open']
high_values = df['high']
low_values = df['low']
close_values = df['close']

# Triple EMA system - Fast, Medium, Slow
df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()   # Fast EMA
df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean() # Medium EMA
df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean() # Slow EMA

# MACD for additional confirmation
df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

# RSI for momentum confirmation
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['close'])

# ATR for dynamic stop loss calculation
def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])

# Plotting setup
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.plot(datetime_objects, close_values, color='black', linestyle='-', alpha=0.8, linewidth=1, label='Price')
plt.plot(datetime_objects, df["EMA9"], linewidth=1, linestyle='-', color='blue', alpha=0.9, label='EMA9 (Fast)')
plt.plot(datetime_objects, df["EMA21"], linewidth=1, linestyle='-', color='orange', alpha=0.9, label='EMA21 (Medium)')
plt.plot(datetime_objects, df["EMA50"], linewidth=1, linestyle='-', color='red', alpha=0.9, label='EMA50 (Slow)')

# Strategy variables
position = "Empty"
entry_target = 0
plot_bar = []
plot_times = []
stoploss = 0
target = 0
entry = 0
trade_open_time = 0
initial_risk = 0
trailing_sl_active = False
best_price = 0  # Track best price for trailing SL

# EMA crossover tracking
prev_ema9 = 0
prev_ema21 = 0
prev_ema50 = 0
prev_macd = 0
prev_macd_signal = 0

# Signal strength tracking
signal_strength = 0
crossover_candles = 0  # Count candles since crossover

for i in range(60, len(datetime_objects)):  # Start after all indicators are stable
    
    current_ema9 = df['EMA9'].iloc[i]
    current_ema21 = df['EMA21'].iloc[i]
    current_ema50 = df['EMA50'].iloc[i]
    current_rsi = df['RSI'].iloc[i]
    current_atr = df['ATR'].iloc[i]
    current_macd = df['MACD'].iloc[i]
    current_macd_signal = df['MACD_Signal'].iloc[i]
    
    if (pd.isna(current_ema9) or pd.isna(current_ema21) or pd.isna(current_ema50) or 
        pd.isna(current_rsi) or pd.isna(current_atr) or pd.isna(current_macd)):
        continue
    
    # Day start reset
    if datetime_objects[i].hour == 9 and datetime_objects[i].minute == 15:
        if print_process:
            print("-----------------------------------\n", datetime_strings[i], "\n-----------------------------------")
        # Reset daily tracking variables
        signal_strength = 0
        crossover_candles = 0

    # Long position management
    if position == "Long":
        current_profit = close_values[i] - entry
        
        # Update best price for trailing
        if close_values[i] > best_price:
            best_price = close_values[i]
        
        # Activate trailing stop when profit >= 1.5x initial risk
        if not trailing_sl_active and current_profit >= (trailing_trigger_ratio * initial_risk):
            trailing_sl_active = True
            if print_process:
                print(f"{datetime_strings[i]} Trailing SL activated at profit: {current_profit:.2f}")
        
        # Update trailing stop loss
        if trailing_sl_active:
            # Trail by moving SL up as price moves favorably
            new_trailing_sl = best_price - (initial_risk * 0.5)  # Keep SL at 50% of initial risk from best price
            if new_trailing_sl > stoploss:
                stoploss = new_trailing_sl
                if print_process:
                    print(f"{datetime_strings[i]} Trailing SL updated to: {stoploss:.2f}")
        
        # Exit conditions
        if ((datetime_objects[i].hour == 15 and datetime_objects[i].minute == 29) or 
            (datetime_objects[i] == (trade_open_time + timedelta(minutes=300)))):  # 5-hour max hold
            
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = close_values[i] - entry
            
            if pnl <= 0:
                plt.subplot(2, 1, 1)
                plt.plot(plot_times, plot_bar, color='r', marker='^', markersize=6, alpha=0.8)
                Ltrade += 1
                Lpoints += pnl
            else:
                plt.subplot(2, 1, 1)
                plt.plot(plot_times, plot_bar, color='g', marker='^', markersize=6, alpha=0.8)
                Ptrade += 1
                Ppoints += pnl
                
            plot_bar.clear()
            plot_times.clear()
            net += pnl
            trailing_sl_active = False
            
            if print_process:
                print(f"{datetime_strings[i]} Long time/EOD exit at {close_values[i]} with {pnl:.2f} points")

        elif close_values[i] <= stoploss:
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = close_values[i] - entry
            
            plt.subplot(2, 1, 1)
            plt.plot(plot_times, plot_bar, color='r', marker='^', markersize=6, alpha=0.8)
            Ltrade += 1
            Lpoints += pnl
            
            plot_bar.clear()
            plot_times.clear()
            net += pnl
            trailing_sl_active = False
            
            exit_type = "Trailing SL" if trailing_sl_active else "Initial SL"
            if print_process:
                print(f"{datetime_strings[i]} Long {exit_type} hit at {close_values[i]} with {pnl:.2f} points")

        elif close_values[i] >= target:
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = close_values[i] - entry
            
            plt.subplot(2, 1, 1)
            plt.plot(plot_times, plot_bar, color='g', marker='^', markersize=6, alpha=0.8)
            Ptrade += 1
            Ppoints += pnl
            
            plot_bar.clear()
            plot_times.clear()
            net += pnl
            trailing_sl_active = False
            
            if print_process:
                print(f"{datetime_strings[i]} Long target hit at {close_values[i]} with {pnl:.2f} points")
        else:
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])

    # Short position management
    if position == "Short":
        current_profit = entry - close_values[i]
        
        # Update best price for trailing (lowest price for short)
        if close_values[i] < best_price:
            best_price = close_values[i]
        
        # Activate trailing stop
        if not trailing_sl_active and current_profit >= (trailing_trigger_ratio * initial_risk):
            trailing_sl_active = True
            if print_process:
                print(f"{datetime_strings[i]} Trailing SL activated at profit: {current_profit:.2f}")
        
        # Update trailing stop loss
        if trailing_sl_active:
            new_trailing_sl = best_price + (initial_risk * 0.5)  # Keep SL at 50% of initial risk from best price
            if new_trailing_sl < stoploss:
                stoploss = new_trailing_sl
                if print_process:
                    print(f"{datetime_strings[i]} Trailing SL updated to: {stoploss:.2f}")
        
        # Exit conditions
        if ((datetime_objects[i].hour == 15 and datetime_objects[i].minute == 29) or 
            (datetime_objects[i] == (trade_open_time + timedelta(minutes=300)))):
            
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = entry - close_values[i]
            
            if pnl <= 0:
                plt.subplot(2, 1, 1)
                plt.plot(plot_times, plot_bar, color='r', marker='v', markersize=6, alpha=0.8)
                Ltrade += 1
                Lpoints += pnl
            else:
                plt.subplot(2, 1, 1)
                plt.plot(plot_times, plot_bar, color='g', marker='v', markersize=6, alpha=0.8)
                Ptrade += 1
                Ppoints += pnl
                
            plot_bar.clear()
            plot_times.clear()
            net += pnl
            trailing_sl_active = False
            
            if print_process:
                print(f"{datetime_strings[i]} Short time/EOD exit at {close_values[i]} with {pnl:.2f} points")

        elif close_values[i] >= stoploss:
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = entry - close_values[i]
            
            plt.subplot(2, 1, 1)
            plt.plot(plot_times, plot_bar, color='r', marker='v', markersize=6, alpha=0.8)
            Ltrade += 1
            Lpoints += pnl
            
            plot_bar.clear()
            plot_times.clear()
            net += pnl
            trailing_sl_active = False
            
            exit_type = "Trailing SL" if trailing_sl_active else "Initial SL"
            if print_process:
                print(f"{datetime_strings[i]} Short {exit_type} hit at {close_values[i]} with {pnl:.2f} points")

        elif close_values[i] <= target:
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = entry - close_values[i]
            
            plt.subplot(2, 1, 1)
            plt.plot(plot_times, plot_bar, color='g', marker='v', markersize=6, alpha=0.8)
            Ptrade += 1
            Ppoints += pnl
            
            plot_bar.clear()
            plot_times.clear()
            net += pnl
            trailing_sl_active = False
            
            if print_process:
                print(f"{datetime_strings[i]} Short target hit at {close_values[i]} with {pnl:.2f} points")
        else:
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])

    # Strategy Logic - 3 EMA Crossover with confirmations
    if (9 <= datetime_objects[i].hour < 15) and position == "Empty":
        
        # Get previous values for crossover detection
        if i > 60:
            prev_ema9 = df['EMA9'].iloc[i-1]
            prev_ema21 = df['EMA21'].iloc[i-1]
            prev_ema50 = df['EMA50'].iloc[i-1]
            prev_macd = df['MACD'].iloc[i-1]
            prev_macd_signal = df['MACD_Signal'].iloc[i-1]
        
        # EMA alignment checks
        bullish_alignment = current_ema9 > current_ema21 > current_ema50
        bearish_alignment = current_ema9 < current_ema21 < current_ema50
        
        # Crossover detection
        fast_cross_medium_up = (current_ema9 > current_ema21) and (prev_ema9 <= prev_ema21)
        fast_cross_medium_down = (current_ema9 < current_ema21) and (prev_ema9 >= prev_ema21)
        
        medium_cross_slow_up = (current_ema21 > current_ema50) and (prev_ema21 <= prev_ema50)
        medium_cross_slow_down = (current_ema21 < current_ema50) and (prev_ema21 >= prev_ema50)
        
        # MACD confirmation
        macd_bullish = current_macd > current_macd_signal and current_macd > 0
        macd_bearish = current_macd < current_macd_signal and current_macd < 0
        macd_cross_up = (current_macd > current_macd_signal) and (prev_macd <= prev_macd_signal)
        macd_cross_down = (current_macd < current_macd_signal) and (prev_macd >= prev_macd_signal)
        
        # RSI momentum confirmation
        rsi_bullish_momentum = 30 < current_rsi < 70 and current_rsi > 50
        rsi_bearish_momentum = 30 < current_rsi < 70 and current_rsi < 50
        
        # Price position relative to EMAs
        price_above_all_emas = close_values[i] > current_ema9 > current_ema21 > current_ema50
        price_below_all_emas = close_values[i] < current_ema9 < current_ema21 < current_ema50
        
        # Calculate signal strength (0-5 scale)
        signal_strength = 0
        
        # LONG ENTRY CONDITIONS
        long_signals = [
            fast_cross_medium_up or medium_cross_slow_up,  # Any EMA crossover up
            bullish_alignment,  # EMA alignment
            macd_bullish or macd_cross_up,  # MACD confirmation
            rsi_bullish_momentum,  # RSI momentum
            price_above_all_emas or close_values[i] > current_ema9  # Price position
        ]
        
        long_signal_strength = sum(long_signals)
        
        # SHORT ENTRY CONDITIONS  
        short_signals = [
            fast_cross_medium_down or medium_cross_slow_down,  # Any EMA crossover down
            bearish_alignment,  # EMA alignment
            macd_bearish or macd_cross_down,  # MACD confirmation
            rsi_bearish_momentum,  # RSI momentum
            price_below_all_emas or close_values[i] < current_ema9  # Price position
        ]
        
        short_signal_strength = sum(short_signals)
        
        # Dynamic ATR-based stop loss calculation
        atr_multiplier = max(current_atr, 15)  # Minimum 15 points
        
        # LONG ENTRY (need at least 3/5 signals)
        if long_signal_strength >= 3:
            position = "Long"
            trade_open_time = datetime_objects[i]
            entry = close_values[i]
            best_price = close_values[i]  # Initialize best price
            
            initial_risk = atr_multiplier * initial_stoploss_multiplier
            stoploss = entry - initial_risk
            target = entry + (initial_risk * initial_target_multiplier)
            trailing_sl_active = False
            
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            
            if print_process:
                print(f"{datetime_strings[i]} LONG ENTRY: {entry}, SL: {stoploss:.2f}, Target: {target:.2f}")
                print(f"Signal Strength: {long_signal_strength}/5, ATR: {atr_multiplier:.2f}")
        
        # SHORT ENTRY (need at least 3/5 signals)
        elif short_signal_strength >= 3:
            position = "Short"
            trade_open_time = datetime_objects[i]
            entry = close_values[i]
            best_price = close_values[i]  # Initialize best price
            
            initial_risk = atr_multiplier * initial_stoploss_multiplier
            stoploss = entry + initial_risk
            target = entry - (initial_risk * initial_target_multiplier)
            trailing_sl_active = False
            
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            
            if print_process:
                print(f"{datetime_strings[i]} SHORT ENTRY: {entry}, SL: {stoploss:.2f}, Target: {target:.2f}")
                print(f"Signal Strength: {short_signal_strength}/5, ATR: {atr_multiplier:.2f}")
    
    # Track net P&L
    net_data.append(net)
    net_data_time.append(datetime_objects[i])

# Plot results
plt.subplot(2, 1, 1)
plt.title('3 EMA Crossover Strategy - Price & EMAs')
plt.xlabel('Datetime')
plt.ylabel('Price')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
plt.plot(net_data_time, net_data, color='purple', linestyle='-', linewidth=2, label='Cumulative P&L')
plt.title('Strategy Performance')
plt.xlabel('Datetime')
plt.ylabel('P&L Points')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()

# Performance metrics
total_trades = Ptrade + Ltrade
print("\n=== 3 EMA CROSSOVER STRATEGY PERFORMANCE ===")
print(f"Net P/L: {net:.2f}")
print(f"Winning Points: {Ppoints:.2f}")
print(f"Losing Points: {Lpoints:.2f}")
print(f"Winning Trades: {Ptrade}")
print(f"Losing Trades: {Ltrade}")
print(f"Total Trades: {total_trades}")

if total_trades > 0:
    print(f"Win Rate: {(Ptrade/total_trades)*100:.2f}%")
    print(f"Average Points per Trade: {net/total_trades:.2f}")
    
    if Ltrade > 0 and Ptrade > 0:
        avg_win = Ppoints/Ptrade
        avg_loss = abs(Lpoints)/Ltrade
        print(f"Average Win: {avg_win:.2f}")
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"Risk:Reward Ratio 1:{avg_win/avg_loss:.2f}")

print(f"Cost per trade: 1")
print("="*50)

# Save data for further analysis
iso_timestamps = [ts.isoformat(timespec='seconds') for ts in datetime_objects]
np.save('ema_crossover_net.npy', np.array(net_data))
np.save('ema_crossover_date.npy', np.array(iso_timestamps))

plt.show()