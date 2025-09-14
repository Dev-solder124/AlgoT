import pandas as pd

# Load the CSV file
df = pd.read_csv("NIFTY50 ML.csv")
print(df)

df['Date'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M').dt.date

# Get sorted unique dates
unique_dates = sorted(df['Date'].unique())

# Initialize train and test date lists using 2:1 pattern
train_dates = []
test_dates = []

i = 0
while i < len(unique_dates):
    train_dates.extend(unique_dates[i:i+2])  # next 2 dates for training
    if i + 2 < len(unique_dates):
        test_dates.append(unique_dates[i+2])  # next 1 date for testing
    i += 3

# Create the datasets based on date membership
train_df = df[df['Date'].isin(train_dates)].drop(columns=['Date']).reset_index(drop=True)
test_df = df[df['Date'].isin(test_dates)].drop(columns=['Date']).reset_index(drop=True)

# Save to CSV files
train_df.to_csv("NIFTY50 TRAIN.csv", index=False)
test_df.to_csv("NIFTY50 TEST.csv", index=False)
print(train_df,'\n',test_df)
print(f"Split complete: {len(train_df)} rows in training set, {len(test_df)} rows in test set.")
