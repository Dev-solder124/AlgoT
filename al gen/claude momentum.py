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

target_multiplier = 2.5  # Risk:Reward = 1:2.5
stoploss_multiplier = 1.0

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

# Technical indicators
df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

# RSI calculation
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['close'])

# Plotting setup
plt.figure(figsize=(12, 8))
plt.plot(datetime_objects, close_values, color='gray', linestyle='-', alpha=0.7)
plt.plot(datetime_objects, df["EMA20"], linewidth=1, linestyle='-', color='blue', alpha=0.8)
plt.plot(datetime_objects, df["EMA50"], linewidth=1, linestyle='-', color='red', alpha=0.8)

# Strategy variables
bar30 = 0  # 30-minute bars (30 x 1-minute bars)
bar30_high = []
bar30_low = []
bar30_open = []
bar30_close = []
bar30_time = []
bar30_volume = []

position = "Empty"
seek_status = "Empty"
entry_target = 0
plot_bar = []
plot_times = []
seek_bar = []
seek_time = []
stoploss = 0
target = 0
entry = 0
trade_open_time = 0

# Breakout tracking
prev_30min_high = 0
prev_30min_low = 0
breakout_strength = 0

# RSI tracking for divergence
rsi_values = []
price_highs = []
price_lows = []

for i in range(50, len(datetime_objects)):  # Start after indicators are stable
    
    current_rsi = df['RSI'].iloc[i]
    current_ema20 = df['EMA20'].iloc[i]
    current_ema50 = df['EMA50'].iloc[i]
    
    if pd.isna(current_rsi) or pd.isna(current_ema20) or pd.isna(current_ema50):
        continue
    
    # Day start reset
    if datetime_objects[i].hour == 9 and datetime_objects[i].minute == 15:
        if print_process:
            print("-----------------------------------\n", datetime_strings[i], "\n-----------------------------------")
        # Reset daily tracking variables
        prev_30min_high = 0
        prev_30min_low = 0
        rsi_values = []
        price_highs = []
        price_lows = []

    # Long position management
    if position == "Long":
        if ((datetime_objects[i].hour == 15 and datetime_objects[i].minute == 29) or 
            (datetime_objects[i] == (trade_open_time + timedelta(minutes=240)))):  # 4-hour max hold
            
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = close_values[i] - entry
            
            if pnl <= 0:
                plt.plot(plot_times, plot_bar, color='r', marker='^', markersize=8)
                Ltrade += 1
                Lpoints += pnl
            else:
                plt.plot(plot_times, plot_bar, color='g', marker='^', markersize=8)
                Ptrade += 1
                Ppoints += pnl
                
            plot_bar.clear()
            plot_times.clear()
            net += pnl
            
            if print_process:
                print(datetime_strings[i], "Long exit at", close_values[i], "with", pnl, "points")

        elif close_values[i] <= stoploss:
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = close_values[i] - entry
            
            plt.plot(plot_times, plot_bar, color='r', marker='^', markersize=8)
            Ltrade += 1
            Lpoints += pnl
            
            plot_bar.clear()
            plot_times.clear()
            net += pnl
            
            if print_process:
                print(datetime_strings[i], "Long SL hit at", close_values[i], "with", pnl, "points")

        elif close_values[i] >= target:
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = close_values[i] - entry
            
            plt.plot(plot_times, plot_bar, color='g', marker='^', markersize=8)
            Ptrade += 1
            Ppoints += pnl
            
            plot_bar.clear()
            plot_times.clear()
            net += pnl
            
            if print_process:
                print(datetime_strings[i], "Long target hit at", close_values[i], "with", pnl, "points")
        else:
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])

    # Short position management
    if position == "Short":
        if ((datetime_objects[i].hour == 15 and datetime_objects[i].minute == 29) or 
            (datetime_objects[i] == (trade_open_time + timedelta(minutes=240)))):
            
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = entry - close_values[i]
            
            if pnl <= 0:
                plt.plot(plot_times, plot_bar, color='r', marker='v', markersize=8)
                Ltrade += 1
                Lpoints += pnl
            else:
                plt.plot(plot_times, plot_bar, color='g', marker='v', markersize=8)
                Ptrade += 1
                Ppoints += pnl
                
            plot_bar.clear()
            plot_times.clear()
            net += pnl
            
            if print_process:
                print(datetime_strings[i], "Short exit at", close_values[i], "with", pnl, "points")

        elif close_values[i] >= stoploss:
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = entry - close_values[i]
            
            plt.plot(plot_times, plot_bar, color='r', marker='v', markersize=8)
            Ltrade += 1
            Lpoints += pnl
            
            plot_bar.clear()
            plot_times.clear()
            net += pnl
            
            if print_process:
                print(datetime_strings[i], "Short SL hit at", close_values[i], "with", pnl, "points")

        elif close_values[i] <= target:
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = entry - close_values[i]
            
            plt.plot(plot_times, plot_bar, color='g', marker='v', markersize=8)
            Ptrade += 1
            Ppoints += pnl
            
            plot_bar.clear()
            plot_times.clear()
            net += pnl
            
            if print_process:
                print(datetime_strings[i], "Short target hit at", close_values[i], "with", pnl, "points")
        else:
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])

    # Build 30-minute bars
    if bar30 == 30:
        curr30_high = max(bar30_high)
        curr30_low = min(bar30_low)
        curr30_close = bar30_close[-1]
        curr30_volume = sum(bar30_volume)
        
        # Strategy logic - only during market hours
        if (9 <= datetime_objects[i].hour < 15) and position == "Empty":
            
            # Condition 1: EMA alignment for trend
            ema_bullish = current_ema20 > current_ema50
            ema_bearish = current_ema20 < current_ema50
            
            # Condition 2: RSI conditions
            rsi_oversold = current_rsi < 35
            rsi_overbought = current_rsi > 65
            rsi_neutral = 35 <= current_rsi <= 65
            
            # Condition 3: Breakout detection
            if prev_30min_high > 0 and prev_30min_low > 0:
                high_breakout = curr30_high > prev_30min_high
                low_breakdown = curr30_low < prev_30min_low
                
                # Calculate breakout strength (how much it broke by)
                breakout_strength = 0
                if high_breakout:
                    breakout_strength = curr30_high - prev_30min_high
                elif low_breakdown:
                    breakout_strength = prev_30min_low - curr30_low
                
                # Only take trades if breakout is significant (> 10 points)
                if breakout_strength > 15:
                    
                    # LONG ENTRY CONDITIONS
                    if (high_breakout and ema_bullish and 
                        (rsi_neutral or (rsi_oversold and current_rsi > 30)) and
                        close_values[i] > current_ema20):
                        
                        position = "Long"
                        trade_open_time = datetime_objects[i]
                        entry = close_values[i]
                        
                        # Dynamic stop loss based on recent volatility
                        atr_proxy = curr30_high - curr30_low
                        if atr_proxy < 5:
                            atr_proxy = 15  # minimum stop distance
                        
                        stoploss = entry - (stoploss_multiplier * atr_proxy)
                        target = entry + (target_multiplier * atr_proxy)
                        
                        plot_bar.append(close_values[i])
                        plot_times.append(datetime_objects[i])
                        
                        if print_process:
                            print(f"{datetime_strings[i]} LONG ENTRY: {entry}, SL: {stoploss:.2f}, Target: {target:.2f}, Breakout: {breakout_strength:.2f}")
                    
                    # SHORT ENTRY CONDITIONS  
                    elif (low_breakdown and ema_bearish and 
                          (rsi_neutral or (rsi_overbought and current_rsi < 70)) and
                          close_values[i] < current_ema20):
                        
                        position = "Short"
                        trade_open_time = datetime_objects[i]
                        entry = close_values[i]
                        
                        # Dynamic stop loss
                        atr_proxy = curr30_high - curr30_low
                        if atr_proxy < 5:
                            atr_proxy = 15
                        
                        stoploss = entry + (stoploss_multiplier * atr_proxy)
                        target = entry - (target_multiplier * atr_proxy)
                        
                        plot_bar.append(close_values[i])
                        plot_times.append(datetime_objects[i])
                        
                        if print_process:
                            print(f"{datetime_strings[i]} SHORT ENTRY: {entry}, SL: {stoploss:.2f}, Target: {target:.2f}, Breakdown: {breakout_strength:.2f}")
        
        # Store current bar as previous for next iteration
        prev_30min_high = curr30_high
        prev_30min_low = curr30_low
        
        # Clear and reset 30min bar data
        bar30_high.clear()
        bar30_low.clear()
        bar30_open.clear()
        bar30_close.clear()
        bar30_time.clear()
        bar30_volume.clear()
        bar30 = 0

    # Add current bar data
    bar30_high.append(high_values[i])
    bar30_low.append(low_values[i])
    bar30_open.append(open_values[i])
    bar30_close.append(close_values[i])
    bar30_time.append(datetime_objects[i])
    bar30_volume.append(0)  # Volume not available in sample
    bar30 += 1
    
    # Track net P&L
    net_data.append(net)
    net_data_time.append(datetime_objects[i])
    
    # End of day cleanup
    if datetime_objects[i].hour == 15 and datetime_objects[i].minute == 29:
        bar30_high.clear()
        bar30_low.clear()
        bar30_open.clear()
        bar30_close.clear()
        bar30_time.clear()
        bar30_volume.clear()
        bar30 = 0

# Plot results
plt.plot(net_data_time, net_data, color='olive', linestyle='-', linewidth=2, label='P&L')
plt.title('Momentum Breakout Strategy - Price & P&L')
plt.xlabel('Datetime')
plt.ylabel('Values')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Performance metrics
total_trades = Ptrade + Ltrade
print("\n=== STRATEGY PERFORMANCE ===")
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
print("="*35)

# Save data for further analysis
iso_timestamps = [ts.isoformat(timespec='seconds') for ts in datetime_objects]
np.save('momentum_net.npy', np.array(net_data))
np.save('momentum_date.npy', np.array(iso_timestamps))

plt.show()