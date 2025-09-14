import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dateutil.parser import parse
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style for light theme
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'gray'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

# Initialize variables
net = 0
Ptrade = 0
Ltrade = 0
n = 375
day = 0

target_multiplier = 2.5
stoploss_multiplier = 1.0

net_data = []
net_data_time = []
trade_markers = []
trade_colors = []
trade_sizes = []

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

# Create subplots with enhanced layout
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1], width_ratios=[3, 1], 
                      hspace=0.3, wspace=0.15)

# Main price chart
ax_main = fig.add_subplot(gs[0, 0])
ax_rsi = fig.add_subplot(gs[1, 0], sharex=ax_main)
ax_pnl = fig.add_subplot(gs[2, 0], sharex=ax_main)
ax_trades = fig.add_subplot(gs[3, 0], sharex=ax_main)

# Stats panel
ax_stats = fig.add_subplot(gs[:, 1])

# Strategy variables
bar30 = 0
bar30_high = []
bar30_low = []
bar30_open = []
bar30_close = []
bar30_time = []
bar30_volume = []

position = "Empty"
entry_target = 0
plot_bar = []
plot_times = []
stoploss = 0
target = 0
entry = 0
trade_open_time = 0

# Enhanced tracking
prev_30min_high = 0
prev_30min_low = 0
breakout_strength = 0
trade_details = []
support_resistance_lines = []
volatility_bands = []

# Daily stats tracking
daily_pnl = []
daily_dates = []
current_day_pnl = 0
last_date = None

for i in range(50, len(datetime_objects)):
    
    current_rsi = df['RSI'].iloc[i]
    current_ema20 = df['EMA20'].iloc[i]
    current_ema50 = df['EMA50'].iloc[i]
    
    if pd.isna(current_rsi) or pd.isna(current_ema20) or pd.isna(current_ema50):
        continue
    
    # Track daily P&L changes
    current_date = datetime_objects[i].date()
    if last_date != current_date:
        if last_date is not None:
            daily_pnl.append(current_day_pnl)
            daily_dates.append(last_date)
        current_day_pnl = 0
        last_date = current_date
    
    # Day start reset
    if datetime_objects[i].hour == 9 and datetime_objects[i].minute == 15:
        if print_process:
            print("-----------------------------------\n", datetime_strings[i], "\n-----------------------------------")
        prev_30min_high = 0
        prev_30min_low = 0

    # Position management with enhanced tracking
    if position == "Long":
        if ((datetime_objects[i].hour == 15 and datetime_objects[i].minute == 29) or 
            (datetime_objects[i] == (trade_open_time + timedelta(minutes=240)))):
            
            position = "Empty"
            pnl = close_values[i] - entry
            exit_reason = "EOD" if datetime_objects[i].hour == 15 else "Time"
            
            # Enhanced trade tracking
            trade_details.append({
                'entry_time': trade_open_time,
                'exit_time': datetime_objects[i],
                'entry_price': entry,
                'exit_price': close_values[i],
                'position': 'Long',
                'pnl': pnl,
                'exit_reason': exit_reason,
                'duration_minutes': (datetime_objects[i] - trade_open_time).total_seconds() / 60
            })
            
            if pnl <= 0:
                trade_markers.append((datetime_objects[i], close_values[i]))
                trade_colors.append('red')
                trade_sizes.append(abs(pnl) * 2 + 50)
                Ltrade += 1
                Lpoints += pnl
            else:
                trade_markers.append((datetime_objects[i], close_values[i]))
                trade_colors.append('green')
                trade_sizes.append(pnl * 2 + 50)
                Ptrade += 1
                Ppoints += pnl
                
            net += pnl
            current_day_pnl += pnl

        elif close_values[i] <= stoploss:
            position = "Empty"
            pnl = close_values[i] - entry
            
            trade_details.append({
                'entry_time': trade_open_time,
                'exit_time': datetime_objects[i],
                'entry_price': entry,
                'exit_price': close_values[i],
                'position': 'Long',
                'pnl': pnl,
                'exit_reason': 'SL',
                'duration_minutes': (datetime_objects[i] - trade_open_time).total_seconds() / 60
            })
            
            trade_markers.append((datetime_objects[i], close_values[i]))
            trade_colors.append('red')
            trade_sizes.append(abs(pnl) * 2 + 50)
            Ltrade += 1
            Lpoints += pnl
            net += pnl
            current_day_pnl += pnl

        elif close_values[i] >= target:
            position = "Empty"
            pnl = close_values[i] - entry
            
            trade_details.append({
                'entry_time': trade_open_time,
                'exit_time': datetime_objects[i],
                'entry_price': entry,
                'exit_price': close_values[i],
                'position': 'Long',
                'pnl': pnl,
                'exit_reason': 'Target',
                'duration_minutes': (datetime_objects[i] - trade_open_time).total_seconds() / 60
            })
            
            trade_markers.append((datetime_objects[i], close_values[i]))
            trade_colors.append('darkgreen')
            trade_sizes.append(pnl * 2 + 50)
            Ptrade += 1
            Ppoints += pnl
            net += pnl
            current_day_pnl += pnl

    # Short position management (similar structure)
    if position == "Short":
        if ((datetime_objects[i].hour == 15 and datetime_objects[i].minute == 29) or 
            (datetime_objects[i] == (trade_open_time + timedelta(minutes=240)))):
            
            position = "Empty"
            pnl = entry - close_values[i]
            exit_reason = "EOD" if datetime_objects[i].hour == 15 else "Time"
            
            trade_details.append({
                'entry_time': trade_open_time,
                'exit_time': datetime_objects[i],
                'entry_price': entry,
                'exit_price': close_values[i],
                'position': 'Short',
                'pnl': pnl,
                'exit_reason': exit_reason,
                'duration_minutes': (datetime_objects[i] - trade_open_time).total_seconds() / 60
            })
            
            if pnl <= 0:
                trade_markers.append((datetime_objects[i], close_values[i]))
                trade_colors.append('red')
                trade_sizes.append(abs(pnl) * 2 + 50)
                Ltrade += 1
                Lpoints += pnl
            else:
                trade_markers.append((datetime_objects[i], close_values[i]))
                trade_colors.append('green')
                trade_sizes.append(pnl * 2 + 50)
                Ptrade += 1
                Ppoints += pnl
                
            net += pnl
            current_day_pnl += pnl

        elif close_values[i] >= stoploss:
            position = "Empty"
            pnl = entry - close_values[i]
            
            trade_details.append({
                'entry_time': trade_open_time,
                'exit_time': datetime_objects[i],
                'entry_price': entry,
                'exit_price': close_values[i],
                'position': 'Short',
                'pnl': pnl,
                'exit_reason': 'SL',
                'duration_minutes': (datetime_objects[i] - trade_open_time).total_seconds() / 60
            })
            
            trade_markers.append((datetime_objects[i], close_values[i]))
            trade_colors.append('red')
            trade_sizes.append(abs(pnl) * 2 + 50)
            Ltrade += 1
            Lpoints += pnl
            net += pnl
            current_day_pnl += pnl

        elif close_values[i] <= target:
            position = "Empty"
            pnl = entry - close_values[i]
            
            trade_details.append({
                'entry_time': trade_open_time,
                'exit_time': datetime_objects[i],
                'entry_price': entry,
                'exit_price': close_values[i],
                'position': 'Short',
                'pnl': pnl,
                'exit_reason': 'Target',
                'duration_minutes': (datetime_objects[i] - trade_open_time).total_seconds() / 60
            })
            
            trade_markers.append((datetime_objects[i], close_values[i]))
            trade_colors.append('darkgreen')
            trade_sizes.append(pnl * 2 + 50)
            Ptrade += 1
            Ppoints += pnl
            net += pnl
            current_day_pnl += pnl

    # Build 30-minute bars and strategy logic
    if bar30 == 30:
        curr30_high = max(bar30_high)
        curr30_low = min(bar30_low)
        curr30_close = bar30_close[-1]
        
        # Add support/resistance levels
        if len(bar30_close) >= 30:
            support_resistance_lines.append((datetime_objects[i], curr30_low, curr30_high))
        
        # Strategy logic
        if (9 <= datetime_objects[i].hour < 15) and position == "Empty":
            
            ema_bullish = current_ema20 > current_ema50
            ema_bearish = current_ema20 < current_ema50
            
            rsi_oversold = current_rsi < 35
            rsi_overbought = current_rsi > 65
            rsi_neutral = 35 <= current_rsi <= 65
            
            if prev_30min_high > 0 and prev_30min_low > 0:
                high_breakout = curr30_high > prev_30min_high
                low_breakdown = curr30_low < prev_30min_low
                
                breakout_strength = 0
                if high_breakout:
                    breakout_strength = curr30_high - prev_30min_high
                elif low_breakdown:
                    breakout_strength = prev_30min_low - curr30_low
                
                if breakout_strength > 10:
                    
                    if (high_breakout and ema_bullish and 
                        (rsi_neutral or (rsi_oversold and current_rsi > 30)) and
                        close_values[i] > current_ema20):
                        
                        position = "Long"
                        trade_open_time = datetime_objects[i]
                        entry = close_values[i]
                        
                        atr_proxy = curr30_high - curr30_low
                        if atr_proxy < 5:
                            atr_proxy = 15
                        
                        stoploss = entry - (stoploss_multiplier * atr_proxy)
                        target = entry + (target_multiplier * atr_proxy)
                        
                        trade_markers.append((datetime_objects[i], entry))
                        trade_colors.append('blue')
                        trade_sizes.append(100)
                        
                        if print_process:
                            print(f"{datetime_strings[i]} LONG ENTRY: {entry}")
                    
                    elif (low_breakdown and ema_bearish and 
                          (rsi_neutral or (rsi_overbought and current_rsi < 70)) and
                          close_values[i] < current_ema20):
                        
                        position = "Short"
                        trade_open_time = datetime_objects[i]
                        entry = close_values[i]
                        
                        atr_proxy = curr30_high - curr30_low
                        if atr_proxy < 5:
                            atr_proxy = 15
                        
                        stoploss = entry + (stoploss_multiplier * atr_proxy)
                        target = entry - (target_multiplier * atr_proxy)
                        
                        trade_markers.append((datetime_objects[i], entry))
                        trade_colors.append('orange')
                        trade_sizes.append(100)
                        
                        if print_process:
                            print(f"{datetime_strings[i]} SHORT ENTRY: {entry}")
        
        prev_30min_high = curr30_high
        prev_30min_low = curr30_low
        
        bar30_high.clear()
        bar30_low.clear()
        bar30_open.clear()
        bar30_close.clear()
        bar30_time.clear()
        bar30_volume.clear()
        bar30 = 0

    bar30_high.append(high_values[i])
    bar30_low.append(low_values[i])
    bar30_open.append(open_values[i])
    bar30_close.append(close_values[i])
    bar30_time.append(datetime_objects[i])
    bar30_volume.append(0)
    bar30 += 1
    
    net_data.append(net)
    net_data_time.append(datetime_objects[i])
    
    if datetime_objects[i].hour == 15 and datetime_objects[i].minute == 29:
        bar30_high.clear()
        bar30_low.clear()
        bar30_open.clear()
        bar30_close.clear()
        bar30_time.clear()
        bar30_volume.clear()
        bar30 = 0

# Enhanced plotting
sample_size = min(len(datetime_objects), 5000)  # Limit points for performance
step = max(1, len(datetime_objects) // sample_size)

# Main price chart with candlestick-like appearance
ax_main.plot(datetime_objects[::step], close_values[::step], 
             color='#2E8B57', linewidth=1.2, alpha=0.8, label='Close Price')
ax_main.plot(datetime_objects[::step], df["EMA20"].iloc[::step], 
             color='#4169E1', linewidth=1.5, alpha=0.9, label='EMA20')
ax_main.plot(datetime_objects[::step], df["EMA50"].iloc[::step], 
             color='#DC143C', linewidth=1.5, alpha=0.9, label='EMA50')

# Add trade markers with different shapes and sizes
if trade_markers:
    times, prices = zip(*trade_markers)
    scatter = ax_main.scatter(times, prices, c=trade_colors, s=trade_sizes, 
                             alpha=0.8, edgecolors='black', linewidth=0.5,
                             zorder=5)

ax_main.set_title('Momentum Breakout Strategy - Enhanced Visualization', 
                  fontsize=14, fontweight='bold', pad=20)
ax_main.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax_main.grid(True, alpha=0.3)
ax_main.set_ylabel('Price', fontsize=12)

# RSI subplot
ax_rsi.plot(datetime_objects[::step], df['RSI'].iloc[::step], 
            color='#9932CC', linewidth=1.5, alpha=0.8)
ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.5)
ax_rsi.axhline(y=30, color='red', linestyle='--', alpha=0.5)
ax_rsi.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
ax_rsi.fill_between(datetime_objects[::step], 70, 100, alpha=0.1, color='red')
ax_rsi.fill_between(datetime_objects[::step], 0, 30, alpha=0.1, color='red')
ax_rsi.set_ylabel('RSI', fontsize=12)
ax_rsi.set_ylim(0, 100)
ax_rsi.grid(True, alpha=0.3)

# P&L curve
ax_pnl.plot(net_data_time[::step], net_data[::step], 
            color='#FFD700', linewidth=2.5, alpha=0.9)
ax_pnl.fill_between(net_data_time[::step], 0, net_data[::step], 
                    alpha=0.3, color='#FFD700')
ax_pnl.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax_pnl.set_ylabel('Cumulative P&L', fontsize=12)
ax_pnl.grid(True, alpha=0.3)

# Trade distribution
trade_hours = [t['entry_time'].hour + t['entry_time'].minute/60 for t in trade_details if t]
if trade_hours:
    ax_trades.hist(trade_hours, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax_trades.set_ylabel('Trade Count', fontsize=12)
    ax_trades.set_xlabel('Hour of Day', fontsize=12)
    ax_trades.grid(True, alpha=0.3)

# Format x-axis
for ax in [ax_main, ax_rsi, ax_pnl, ax_trades]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(set([d.date() for d in datetime_objects])) // 10)))

# Statistics panel
ax_stats.axis('off')
total_trades = Ptrade + Ltrade

stats_text = f"""
╔═══ STRATEGY PERFORMANCE ═══╗
║                                                    ║
║  Net P&L: {net:.2f}                          ║
║  Total Trades: {total_trades}                     ║
║  Winning Trades: {Ptrade}                    ║
║  Losing Trades: {Ltrade}                      ║
║                                                    ║
"""

if total_trades > 0:
    win_rate = (Ptrade/total_trades)*100
    avg_trade = net/total_trades
    stats_text += f"""║  Win Rate: {win_rate:.1f}%                   ║
║  Avg P&L/Trade: {avg_trade:.2f}            ║
"""

if Ltrade > 0 and Ptrade > 0:
    avg_win = Ppoints/Ptrade
    avg_loss = abs(Lpoints)/Ltrade
    risk_reward = avg_win/avg_loss
    stats_text += f"""║                                                    ║
║  Average Win: {avg_win:.2f}                ║
║  Average Loss: {avg_loss:.2f}               ║
║  Risk:Reward: 1:{risk_reward:.2f}        ║
"""

stats_text += f"""║                                                    ║
║  Profit Factor: {abs(Ppoints/Lpoints) if Lpoints != 0 else 'N/A':.2f}           ║
║  Max Trades/Day: {max([daily_pnl.count(d) for d in set(daily_dates)]) if daily_dates else 0}              ║
║                                                    ║
╚══════════════════════════╝

Strategy Details:
• 30-min breakout detection
• EMA trend confirmation  
• RSI momentum filter
• Dynamic stop-loss sizing
• 1:2.5 risk-reward ratio
"""

ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
              fontsize=11, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

plt.suptitle('Advanced Momentum Strategy Dashboard', fontsize=16, fontweight='bold', y=0.98)

# Performance metrics output
total_trades = Ptrade + Ltrade
print("\n" + "="*50)
print("       ENHANCED STRATEGY PERFORMANCE")
print("="*50)
print(f"Net P&L: {net:.2f}")
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {(Ptrade/total_trades)*100:.2f}%" if total_trades > 0 else "No trades")
print(f"Avg Trade Duration: {np.mean([t['duration_minutes'] for t in trade_details]):.1f} minutes" if trade_details else "N/A")

# Save enhanced data
iso_timestamps = [ts.isoformat(timespec='seconds') for ts in datetime_objects]
np.save('enhanced_momentum_net.npy', np.array(net_data))
np.save('enhanced_momentum_date.npy', np.array(iso_timestamps))

# Save trade details
trade_df = pd.DataFrame(trade_details)
if not trade_df.empty:
    trade_df.to_csv('trade_details.csv', index=False)

plt.tight_layout()
plt.show()