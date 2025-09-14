import csv

def exponential_moving_average(data, window):
    alpha = 2 / (window + 1)
    ema = [data[0]]  # Initialize with the first data point
    for i in range(1, len(data)):
        ema_value = alpha * data[i] + (1 - alpha) * ema[i-1]
        ema.append(ema_value)
    return ema

# Read the CSV file
filename = 'NIFTY 50 COPY.csv'
timestamps = []
open_prices = []
high_prices = []
low_prices = []
close_prices = []

with open(filename, 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Skip header
    for row in reader:
        timestamps.append(row[0])
        open_prices.append(float(row[1]))
        high_prices.append(float(row[2]))
        low_prices.append(float(row[3]))
        close_prices.append(float(row[4]))

# Compute EMAs
ema10 = exponential_moving_average(close_prices, 10)
ema50 = exponential_moving_average(close_prices, 50)

# Trading parameters
target_increment = 10
stoploss_increment = 1

# Initialize trading variables
position = None  # 'long', 'short', or None
entry_price = 0
target = 0
stoploss = 0

Ppoints = 0
Lpoints = 0
Ptrade = 0
Ltrade = 0

# Main loop to check for signals and manage trades
for i in range(1, len(close_prices)):
    if position is None:
        # Check for bullish crossover (EMA10 crosses above EMA50)
        if ema10[i] > ema50[i] and ema10[i-1] <= ema50[i-1]:
            position = 'long'
            entry_price = close_prices[i]
            target = entry_price + target_increment
            stoploss = entry_price - stoploss_increment
    else:
        current_high = high_prices[i]
        current_low = low_prices[i]
        if position == 'long':
            # Check for target hit
            if current_high >= target:
                Ppoints += target_increment
                Ptrade += 1
                position = None
            # Check for stoploss hit
            elif current_low <= stoploss:
                Lpoints += stoploss_increment
                Ltrade += 1
                position = None

# Calculate net points
net_points = Ppoints - Lpoints

# Output results
print("Net Points Captured:", net_points)
print("Profitable Points (Ppoints):", Ppoints)
print("Loss Points (Lpoints):", Lpoints)
print("Profitable Trades (Ptrade):", Ptrade)
print("Loss Trades (Ltrade):", Ltrade)
print("Total Trades:", Ptrade + Ltrade)
if Ltrade != 0:
    print(f"Win Ratio: {Ptrade / (Ptrade + Ltrade):.2f}")
    print(f"Profit Factor: {Ppoints / Lpoints:.2f}")