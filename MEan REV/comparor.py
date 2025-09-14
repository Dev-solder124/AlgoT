import pandas as pd
import matplotlib.pyplot as plt

# === EDIT THESE FILE PATHS ===
bse_file = r"D:\AlgoT\HYUNDAIBSE.csv"
nse_file = r"D:\AlgoT\HYUNDAINSE.csv"

# Read CSVs
def read_stock_data(filepath):
    df = pd.read_csv(filepath, header=None, names=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df

bse_df = read_stock_data(bse_file)
nse_df = read_stock_data(nse_file)

# Sort dataframes by timestamp
bse_df = bse_df.sort_values("timestamp")
nse_df = nse_df.sort_values("timestamp")

# Merge to align each NSE timestamp to the most recent BSE timestamp not after it
merged_df = pd.merge_asof(
    nse_df,
    bse_df,
    on="timestamp",
    direction="backward",
    suffixes=('_nse', '_bse')
)

# Drop rows where there was no matching BSE timestamp
merged_df = merged_df.dropna(subset=["price_bse"])

# Compute price difference
merged_df["price_diff"] = (merged_df["price_nse"] - merged_df["price_bse"])

# Plotting
plt.figure(figsize=(14, 7))

plt.plot(merged_df["timestamp"], merged_df["price_bse"], label="BSE (aligned)", color="blue")
plt.plot(merged_df["timestamp"], merged_df["price_nse"], label="NSE", color="green")
# plt.plot(merged_df["timestamp"], merged_df["price_diff"], label="NSE - BSE", color="red")

plt.xlabel("Time")
plt.ylabel("Price / Difference")
plt.title("NSE vs BSE Price Comparison with Difference (Aligned by Latest BSE Timestamp)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
