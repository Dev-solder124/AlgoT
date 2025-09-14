import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dateutil.parser import parse
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec

# Set light theme
plt.style.use('default')  # Light theme
sns.set_palette("Set1")

net = 0
Ptrade = 0
Ltrade = 0
rown = []
n = 375
day = 0

start_day = 1 + (day * n)

target_increment = 10
stoploss_increment = 1

net_data = []
net_data_time = []
trade_history = []  # Store trade details for analysis

Ppoints = 0
Lpoints = 0

print_processs = 1

df = pd.read_csv(r"D:\AlgoT\NIFTY 50 COPY.csv", usecols=range(6))

datetime_strings = df['date']

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

data = []
dat_element = []
alert_info = []

# Calculate EMA
df['EMA75'] = df['close'].ewm(span=75, adjust=False).mean()

# Variables for bar aggregation
bar15 = 0
bar5 = 0
bar15_high = []
bar5_high = []
bar15_low = []
bar5_low = []
bar15_open = []
bar5_open = []
bar15_close = []
bar5_close = []
bar15_time = []
bar5_time = []
bar15_closevalue = 0
curr15_high = close_values[0]
curr15_low = close_values[0]
curr5_high = close_values[0]
curr5_low = close_values[0]

position = "Empty"
seek_status = "Empty"
entry_target = 0
plot_bar = []
plot_times = []
seek_bar = []
seek_time = []
temp_stoploss = 0
stoploss = 0
target = 0
buffer = 0
trtbuffer = 0
entry = 0
gap = 0

trade_open_time = 0

real_ema = []
real_ema_time = []
previous_ema = close_values[0]

window = 75
alpha = 2 / (window + 1)

# Storage for visualization data
long_entries = []
long_exits = []
short_entries = []
short_exits = []
profitable_trades = []
losing_trades = []

for i in range(0, len(datetime_objects)):
    
    ema_value = alpha * close_values[i] + (1 - alpha) * previous_ema
    previous_ema = ema_value

    real_ema.append(ema_value)
    real_ema_time.append(datetime_objects[i])

    if datetime_objects[i].hour == 9 and datetime_objects[i].minute == 15:
        if print_processs:
            print("-----------------------------------\n", datetime_strings[i], "\n-----------------------------------")

    if position == "Long":
        if ((datetime_objects[i].hour == 15 and datetime_objects[i].minute == 29) or 
            (datetime_objects[i] == (trade_open_time + timedelta(minutes=0)))):
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = close_values[i] - entry
            
            trade_data = {
                'type': 'Long',
                'entry_time': plot_times[0],
                'exit_time': datetime_objects[i],
                'entry_price': entry,
                'exit_price': close_values[i],
                'pnl': pnl,
                'exit_reason': 'EOD'
            }
            trade_history.append(trade_data)
            
            if pnl <= 0:
                long_exits.append((datetime_objects[i], close_values[i], 'loss'))
                Ltrade += 1
                Lpoints += pnl
                losing_trades.append(trade_data)
            else:
                long_exits.append((datetime_objects[i], close_values[i], 'profit'))
                Ptrade += 1
                Ppoints += pnl
                profitable_trades.append(trade_data)
            
            plot_bar.clear()
            plot_times.clear()
            if print_processs:
                print(datetime_strings[i], "Long exit at", close_values[i], "with", pnl, "points")
            net += pnl

        elif close_values[i] <= stoploss:
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = close_values[i] - entry
            
            trade_data = {
                'type': 'Long',
                'entry_time': plot_times[0],
                'exit_time': datetime_objects[i],
                'entry_price': entry,
                'exit_price': close_values[i],
                'pnl': pnl,
                'exit_reason': 'Stoploss'
            }
            trade_history.append(trade_data)
            
            if pnl <= 0:
                long_exits.append((datetime_objects[i], close_values[i], 'loss'))
                Ltrade += 1
                Lpoints += pnl
                losing_trades.append(trade_data)
            else:
                long_exits.append((datetime_objects[i], close_values[i], 'profit'))
                Ptrade += 1
                Ppoints += pnl
                profitable_trades.append(trade_data)
            
            plot_bar.clear()
            plot_times.clear()
            if print_processs:
                print(datetime_strings[i], "Long exit at", close_values[i], "with", pnl, "points")
            net += pnl

        elif close_values[i] >= target:
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = close_values[i] - entry
            
            trade_data = {
                'type': 'Long',
                'entry_time': plot_times[0],
                'exit_time': datetime_objects[i],
                'entry_price': entry,
                'exit_price': close_values[i],
                'pnl': pnl,
                'exit_reason': 'Target'
            }
            trade_history.append(trade_data)
            
            long_exits.append((datetime_objects[i], close_values[i], 'profit'))
            plot_bar.clear()
            plot_times.clear()
            if print_processs:
                print(datetime_strings[i], "Long exit at", close_values[i], "with", pnl, "points")
            net += pnl
            Ptrade += 1
            Ppoints += pnl
            profitable_trades.append(trade_data)

        else:
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])

    if position == "Short":
        if ((datetime_objects[i].hour == 15 and datetime_objects[i].minute == 29) or 
            (datetime_objects[i] == (trade_open_time + timedelta(minutes=0)))):
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = -close_values[i] + entry
            
            trade_data = {
                'type': 'Short',
                'entry_time': plot_times[0],
                'exit_time': datetime_objects[i],
                'entry_price': entry,
                'exit_price': close_values[i],
                'pnl': pnl,
                'exit_reason': 'EOD'
            }
            trade_history.append(trade_data)
            
            if pnl <= 0:
                short_exits.append((datetime_objects[i], close_values[i], 'loss'))
                Ltrade += 1
                Lpoints += pnl
                losing_trades.append(trade_data)
            else:
                short_exits.append((datetime_objects[i], close_values[i], 'profit'))
                Ptrade += 1
                Ppoints += pnl
                profitable_trades.append(trade_data)
            
            plot_bar.clear()
            plot_times.clear()
            if print_processs:
                print(datetime_strings[i], "Short exit at", close_values[i], "with", pnl, "points")
            net += pnl

        elif close_values[i] >= stoploss:
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = -close_values[i] + entry
            
            trade_data = {
                'type': 'Short',
                'entry_time': plot_times[0],
                'exit_time': datetime_objects[i],
                'entry_price': entry,
                'exit_price': close_values[i],
                'pnl': pnl,
                'exit_reason': 'Stoploss'
            }
            trade_history.append(trade_data)
            
            if pnl <= 0:
                short_exits.append((datetime_objects[i], close_values[i], 'loss'))
                Ltrade += 1
                Lpoints += pnl
                losing_trades.append(trade_data)
            else:
                short_exits.append((datetime_objects[i], close_values[i], 'profit'))
                Ptrade += 1
                Ppoints += pnl
                profitable_trades.append(trade_data)
            
            plot_bar.clear()
            plot_times.clear()
            if print_processs:
                print(datetime_strings[i], "Short exit at", close_values[i], "with", pnl, "points")
            net += pnl

        elif close_values[i] <= target:
            position = "Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            pnl = -close_values[i] + entry
            
            trade_data = {
                'type': 'Short',
                'entry_time': plot_times[0],
                'exit_time': datetime_objects[i],
                'entry_price': entry,
                'exit_price': close_values[i],
                'pnl': pnl,
                'exit_reason': 'Target'
            }
            trade_history.append(trade_data)
            
            short_exits.append((datetime_objects[i], close_values[i], 'profit'))
            plot_bar.clear()
            plot_times.clear()
            if print_processs:
                print(datetime_strings[i], "Short exit at", close_values[i], "with", pnl, "points")
            net += pnl
            Ptrade += 1
            Ppoints += pnl
            profitable_trades.append(trade_data)

        else:
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])

    # [Rest of the original logic for bar aggregation and signal generation]
    if bar5 == 15:
        curr5_high = max(bar5_high)
        curr5_low = min(bar5_low)
        bar5_closevalue = bar5_close[-1]

        if seek_status == "Long seeking" and position == "Empty":
            seek_time.extend(bar5_time)
            seek_bar.extend(bar5_close)
            if close_values[i] >= entry_target:
                position = "Long"
                long_entries.append((datetime_objects[i], close_values[i]))
                
                seek_status = "Empty"
                plot_bar.append(close_values[i])
                plot_times.append(datetime_objects[i])
                entry = close_values[i]
                
                buffer = abs(entry_target - temp_stoploss)
                trtbuffer = buffer
                if buffer == 0:
                    buffer = 0.5
                stoploss = entry - (stoploss_increment * buffer)
                target = entry + target_increment * trtbuffer
                if print_processs:
                    print(datetime_strings[i], position, "entry=", entry, "buffer:", buffer, "SL", stoploss, "Tgrt", target)

                seek_bar = []
                seek_time = []

            elif close_values[i] <= temp_stoploss:
                if print_processs:
                    print("Long abandoned, Short taken")
                position = "Short"
                short_entries.append((datetime_objects[i], close_values[i]))
                
                seek_status = "Empty"
                plot_bar.append(close_values[i])
                plot_times.append(datetime_objects[i])
                entry = close_values[i]
                buffer = abs(-temp_stoploss + entry_target)
                trtbuffer = buffer
                if buffer == 0:
                    buffer = 0.5
                target = entry - target_increment * trtbuffer
                stoploss = entry + stoploss_increment * buffer
                if print_processs:
                    print(datetime_strings[i], position, "entry=", entry, "buffer:", buffer, "SL", stoploss, "Tgrt", target)

                seek_bar = []
                seek_time = []
            
            else:
                seek_status = "Empty"
                dat_element = []

        if seek_status == "Short seeking" and position == "Empty":
            seek_time.extend(bar5_time)
            seek_bar.extend(bar5_close)
            if close_values[i] <= entry_target:
                position = "Short"
                short_entries.append((datetime_objects[i], close_values[i]))
                
                seek_status = "Empty"
                plot_bar.append(close_values[i])
                plot_times.append(datetime_objects[i])
                entry = close_values[i]
                stoploss = temp_stoploss
                buffer = abs(temp_stoploss - entry_target)
                trtbuffer = buffer
                if buffer == 0:
                    buffer = 0.5
                target = entry - target_increment * trtbuffer
                stoploss = entry + stoploss_increment * buffer

                if print_processs:
                    print(datetime_strings[i], position, "entry=", entry, "buffer:", buffer, "SL", stoploss, "Tgrt", target)

                seek_bar = []
                seek_time = []

            elif close_values[i] >= temp_stoploss:
                if print_processs:
                    print("Short abandoned, Long taken")
                position = "Long"
                long_entries.append((datetime_objects[i], close_values[i]))
                
                seek_status = "Empty"
                plot_bar.append(close_values[i])
                plot_times.append(datetime_objects[i])
                entry = close_values[i]
                buffer = abs(temp_stoploss - entry_target)
                trtbuffer = buffer
                stoploss = temp_stoploss
                if buffer == 0:
                    buffer = 0.5
                target = entry + target_increment * trtbuffer
                stoploss = entry - stoploss_increment * buffer

                if print_processs:
                    print(datetime_strings[i], position, "entry=", entry, "buffer:", buffer, "SL", stoploss, "Tgrt", target)

                seek_bar = []
                seek_time = []
            
            else:
                seek_status = "Empty"
                dat_element = []

        bar5_high.clear()
        bar5_low.clear()
        bar5_open.clear()
        bar5_close.clear()
        bar5_time.clear()

        bar5 = 0
        
        bar5_high.append(high_values[i])
        bar5_low.append(low_values[i])
        bar5_open.append(open_values[i])
        bar5_close.append(close_values[i])
        bar5_time.append(datetime_objects[i])
        bar5 += 1
    else:
        bar5_high.append(high_values[i])
        bar5_low.append(low_values[i])
        bar5_open.append(open_values[i])
        bar5_close.append(close_values[i])
        bar5_time.append(datetime_objects[i])
        bar5 += 1

    if bar15 == 15:
        curr15_high = max(bar15_high)
        curr15_low = min(bar15_low)
        bar15_closevalue = bar15_close[-1]

        cdl_body = curr15_high - curr15_low

        if seek_status == "Empty":
            if curr15_low > ema_value:
                seek_status = "Short seeking"
                trade_open_time = datetime_objects[i]
                entry_target = curr15_low
                gap = curr15_low - ema_value
                temp_stoploss = curr15_high
                seek_bar.clear()
                seek_time.clear()
                seek_bar.append(close_values[i])
                seek_time.append(datetime_objects[i])

                if print_processs:
                    print(datetime_strings[i], seek_status)

            elif curr15_high < ema_value:
                seek_status = "Long seeking"
                trade_open_time = datetime_objects[i]
                entry_target = curr15_high
                gap = ema_value - curr15_high
                temp_stoploss = curr15_low
                
                seek_bar.clear()
                seek_time.clear()
                seek_bar.append(close_values[i])
                seek_time.append(datetime_objects[i])

                if print_processs:
                    print(datetime_strings[i], seek_status)

        bar15_high.clear()
        bar15_low.clear()
        bar15_open.clear()
        bar15_close.clear()
        bar15_time.clear()

        bar15 = 0
        
        bar15_high.append(high_values[i])
        bar15_low.append(low_values[i])
        bar15_open.append(open_values[i])
        bar15_close.append(close_values[i])
        bar15_time.append(datetime_objects[i])
        bar15 += 1
    else:
        bar15_high.append(high_values[i])
        bar15_low.append(low_values[i])
        bar15_open.append(open_values[i])
        bar15_close.append(close_values[i])
        bar15_time.append(datetime_objects[i])
        bar15 += 1
        
    net_data.append(net)
    net_data_time.append(datetime_objects[i])

    if datetime_objects[i].hour == 15 and datetime_objects[i].minute == 29:
        seek_bar = []
        seek_time = []
        if seek_status == "Long seeking" or seek_status == "Short seeking":
            seek_status = "Empty"
            dat_element = []
        bar15_high.clear()
        bar15_low.clear()
        bar15_open.clear()
        bar15_close.clear()
        bar15_time.clear()
        bar15 = 0
        bar5_high.clear()
        bar5_low.clear()
        bar5_open.clear()
        bar5_close.clear()
        bar5_time.clear()
        bar5 = 0

# Calculate statistics
total_trades = Ptrade + Ltrade
win_rate = (Ptrade / total_trades * 100) if total_trades > 0 else 0
avg_win = (Ppoints / Ptrade) if Ptrade > 0 else 0
avg_loss = (Lpoints / Ltrade) if Ltrade > 0 else 0
risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
profit_factor = abs(Ppoints / Lpoints) if Lpoints != 0 else float('inf')
max_drawdown = min(net_data) if net_data else 0

# Prepare optimized data for plotting (sample every 5th point for better performance)
sample_step = 5
sample_idx = range(0, len(datetime_objects), sample_step)
sample_times = [datetime_objects[i] for i in sample_idx]
sample_close = [close_values[i] for i in sample_idx]
sample_ema = [real_ema[i] for i in sample_idx]
sample_net_times = [net_data_time[i] for i in sample_idx if i < len(net_data_time)]
sample_net_data = [net_data[i] for i in sample_idx if i < len(net_data)]

class NavigationHandler:
    def __init__(self):
        self.current_fig = 0
        self.figures = []
        
    def create_figure_1(self):
        """Price Chart with Trade Signals"""
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.suptitle('Price Chart with Trade Signals', fontsize=16, fontweight='bold')
        
        # Plot price and EMA with full data for detailed analysis
        ax.plot(datetime_objects, close_values, color='#2E86C1', linewidth=0.8, alpha=0.8, label='Close Price')
        ax.plot(real_ema_time, real_ema, color='#E67E22', linewidth=2, label='EMA 75', alpha=0.9)
        
        # Plot trade signals
        if long_entries:
            entry_times, entry_prices = zip(*long_entries)
            ax.scatter(entry_times, entry_prices, color='#27AE60', marker='^', s=60, 
                      zorder=5, alpha=0.9, label='Long Entry', edgecolors='darkgreen')
        
        if short_entries:
            entry_times, entry_prices = zip(*short_entries)
            ax.scatter(entry_times, entry_prices, color='#E74C3C', marker='v', s=60, 
                      zorder=5, alpha=0.9, label='Short Entry', edgecolors='darkred')
        
        # Plot exits
        for time, price, result in long_exits:
            color = '#27AE60' if result == 'profit' else '#E74C3C'
            ax.scatter(time, price, color=color, marker='x', s=40, zorder=5, alpha=0.8)
        
        for time, price, result in short_exits:
            color = '#27AE60' if result == 'profit' else '#E74C3C'
            ax.scatter(time, price, color=color, marker='x', s=40, zorder=5, alpha=0.8)
        
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Format x-axis for better date display
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        return fig
        
    def create_figure_2(self):
        """P&L Curve"""
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.suptitle('Cumulative P&L Curve', fontsize=16, fontweight='bold')
        
        ax.plot(net_data_time, net_data, color='#F39C12', linewidth=2.5)
        ax.fill_between(net_data_time, net_data, alpha=0.3, color='#F39C12')
        ax.axhline(y=0, color='#34495E', linestyle='-', alpha=0.7, linewidth=1)
        
        ax.set_ylabel('Cumulative P&L', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add statistics text box
        stats_text = f'Final P&L: {net:.2f}\nTotal Trades: {total_trades}\nWin Rate: {win_rate:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        return fig
        
    def create_figure_3(self):
        """Trade Analysis"""
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.suptitle('Individual Trade P&L Analysis', fontsize=16, fontweight='bold')
        
        if trade_history:
            trade_pnls = [trade['pnl'] for trade in trade_history]
            colors = ['#27AE60' if pnl > 0 else '#E74C3C' for pnl in trade_pnls]
            bars = ax.bar(range(len(trade_pnls)), trade_pnls, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axhline(y=0, color='#34495E', linestyle='-', alpha=0.7, linewidth=1)
            
            ax.set_xlabel('Trade Number', fontsize=12)
            ax.set_ylabel('P&L', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on significant bars
            max_pnl = max(trade_pnls) if trade_pnls else 0
            min_pnl = min(trade_pnls) if trade_pnls else 0
            threshold = max(abs(max_pnl), abs(min_pnl)) * 0.5
            
            for i, (bar, pnl) in enumerate(zip(bars, trade_pnls)):
                if abs(pnl) >= threshold:  # Label only significant trades
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (2 if height >= 0 else -5),
                           f'{pnl:.1f}', ha='center', va='bottom' if height >= 0 else 'top', 
                           fontsize=8, fontweight='bold')
            
            # Statistics text
            stats_text = f'Best Trade: {max_pnl:.2f}\nWorst Trade: {min_pnl:.2f}\nAvg Win: {avg_win:.2f}\nAvg Loss: {avg_loss:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
        
    def create_figure_4(self):
        """Monthly Performance"""
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.suptitle('Monthly P&L Distribution', fontsize=16, fontweight='bold')
        
        if trade_history:
            # Group trades by month
            monthly_pnl = {}
            monthly_trades = {}
            for trade in trade_history:
                month_key = trade['entry_time'].strftime('%Y-%m')
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0
                    monthly_trades[month_key] = 0
                monthly_pnl[month_key] += trade['pnl']
                monthly_trades[month_key] += 1
            
            months = list(monthly_pnl.keys())
            pnls = list(monthly_pnl.values())
            colors = ['#27AE60' if pnl > 0 else '#E74C3C' for pnl in pnls]
            
            bars = ax.bar(months, pnls, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axhline(y=0, color='#34495E', linestyle='-', alpha=0.7, linewidth=1)
            
            ax.set_ylabel('Monthly P&L', fontsize=12)
            ax.set_xlabel('Month', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value and trade count labels
            for bar, pnl, month in zip(bars, pnls, months):
                height = bar.get_height()
                trades_count = monthly_trades[month]
                ax.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -15),
                       f'{pnl:.1f}\n({trades_count}T)', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        return fig
        
    def create_figure_5(self):
        """Statistics Dashboard"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Trading Strategy Statistics', fontsize=16, fontweight='bold')
        
        # Remove axes and add text
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Calculate additional statistics
        if trade_history:
            exit_reasons = {}
            for trade in trade_history:
                reason = trade['exit_reason']
                if reason not in exit_reasons:
                    exit_reasons[reason] = {'count': 0, 'pnl': 0}
                exit_reasons[reason]['count'] += 1
                exit_reasons[reason]['pnl'] += trade['pnl']
            
            # Consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_streak = 0
            current_type = None
            
            for trade in trade_history:
                if trade['pnl'] > 0:
                    if current_type == 'win':
                        current_streak += 1
                    else:
                        current_streak = 1
                        current_type = 'win'
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                else:
                    if current_type == 'loss':
                        current_streak += 1
                    else:
                        current_streak = 1
                        current_type = 'loss'
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)
        
        # Create comprehensive statistics text
        stats_text = f"""
TRADING STRATEGY PERFORMANCE REPORT
{'='*60}

📊 OVERALL PERFORMANCE
    Total P&L:                     {net:>12.2f}
    Total Trades:                  {total_trades:>12}
    Average per Trade:             {net/total_trades if total_trades > 0 else 0:>12.2f}
    
💹 WIN/LOSS BREAKDOWN
    Winning Trades:                {Ptrade:>12}
    Losing Trades:                 {Ltrade:>12}
    Win Rate:                      {win_rate:>11.1f}%
    
💰 PROFIT/LOSS METRICS
    Total Profit Points:           {Ppoints:>12.2f}
    Total Loss Points:             {Lpoints:>12.2f}
    Average Win:                   {avg_win:>12.2f}
    Average Loss:                  {avg_loss:>12.2f}
    
📈 RISK MANAGEMENT
    Risk:Reward Ratio:             1:{risk_reward:>9.2f}
    Profit Factor:                 {profit_factor:>12.2f}
    Maximum Drawdown:              {max_drawdown:>12.2f}
    
🎯 EXTREMES
    Best Trade:                    {max([t['pnl'] for t in trade_history]) if trade_history else 0:>12.2f}
    Worst Trade:                   {min([t['pnl'] for t in trade_history]) if trade_history else 0:>12.2f}
    
📅 TRADING PERIOD
    Start Date:                    {datetime_objects[0].strftime('%Y-%m-%d') if datetime_objects else 'N/A':>12}
    End Date:                      {datetime_objects[-1].strftime('%Y-%m-%d') if datetime_objects else 'N/A':>12}
    Total Days:                    {(datetime_objects[-1] - datetime_objects[0]).days if len(datetime_objects) > 1 else 0:>12}

🔧 STRATEGY PARAMETERS
    Target Increment:              {target_increment:>12}
    Stop Loss Increment:           {stoploss_increment:>12}
    EMA Period:                    {window:>12}
"""

        if trade_history:
            stats_text += f"""
🎲 CONSECUTIVE TRADES
    Max Consecutive Wins:          {max_consecutive_wins:>12}
    Max Consecutive Losses:        {max_consecutive_losses:>12}

🚪 EXIT REASON ANALYSIS
"""
            for reason, data in exit_reasons.items():
                avg_pnl = data['pnl'] / data['count'] if data['count'] > 0 else 0
                stats_text += f"    {reason:<15}:          {data['count']:>3} trades, {data['pnl']:>8.2f} total, {avg_pnl:>6.2f} avg\n"

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.9, pad=1))
        
        return fig

    def show_all_figures(self):
        """Create and show all figures"""
        self.figures = [
            self.create_figure_1(),  # Price Chart
            self.create_figure_2(),  # P&L Curve  
            self.create_figure_3(),  # Trade Analysis
            self.create_figure_4(),  # Monthly Performance
            self.create_figure_5()   # Statistics
        ]
        
        # Add navigation text to each figure
        nav_text = "Press 'n' for next chart, 'p' for previous chart, 'q' to quit"
        for i, fig in enumerate(self.figures):
            fig.text(0.5, 0.02, f"Chart {i+1}/5 - {nav_text}", 
                    ha='center', va='bottom', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Show first figure
        self.current_fig = 0
        self.figures[self.current_fig].show()
        
        # Set up key press handler
        for fig in self.figures:
            fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    def on_key_press(self, event):
        """Handle key press events for navigation"""
        if event.key == 'n':  # Next
            self.current_fig = (self.current_fig + 1) % len(self.figures)
            self.figures[self.current_fig].show()
            plt.figure(self.figures[self.current_fig].number)
        elif event.key == 'p':  # Previous
            self.current_fig = (self.current_fig - 1) % len(self.figures)
            self.figures[self.current_fig].show()
            plt.figure(self.figures[self.current_fig].number)
        elif event.key == 'q':  # Quit
            plt.close('all')

# Create navigation handler and show figures
nav = NavigationHandler()
nav.show_all_figures()

# Print console statistics
print("=" * 60)
print("TRADING STRATEGY RESULTS")
print("=" * 60)
print(f"Net P/L: {net}")
print(f"Profit Points: {Ppoints}")
print(f"Loss Points: {Lpoints}")
print(f"Profitable Trades: {Ptrade}")
print(f"Losing Trades: {Ltrade}")
if total_trades > 0:
    print(f"Average points per trade: {net/total_trades:.2f}")
print(f"Cost per trade: 1")
print(f"Total number of trades: {total_trades}")
if Ltrade != 0:
    print(f"Win Percentage: {win_rate:.1f}%")
if Ptrade > 0 and Ltrade > 0:
    print(f"Average Risk to Reward Ratio 1: {risk_reward:.2f}")

print("\n" + "=" * 60)
print("NAVIGATION INSTRUCTIONS")
print("=" * 60)
print("• Press 'n' to go to next chart")
print("• Press 'p' to go to previous chart") 
print("• Press 'q' to quit all charts")
print("• Use matplotlib's zoom, pan, and other tools normally")
print("• Each chart is fully interactive with detailed data")

# Save data
iso_timestamps = []
for ts in datetime_objects:
    dt = ts
    iso_timestamps.append(dt.isoformat(timespec='seconds'))

np.save('net.npy', np.array(net_data))
np.save('date.npy', np.array(iso_timestamps))

plt.show()