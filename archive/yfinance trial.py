import yfinance as yf

# Fetch Nifty 50 data
nifty_data = yf.download("^NSEI", start="2025-02-08", end="2025-02-12")

print(nifty_data.head())