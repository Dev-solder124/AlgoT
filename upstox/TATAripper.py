import requests
import pandas as pd
from datetime import datetime

# Replace with your actual credentials
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIzRENQWlUiLCJqdGkiOiI2N2ViYjk0ZmE4ZTMwMzUyMTRlMThlYWIiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaWF0IjoxNzQzNTAxNjQ3LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NDM1NDQ4MDB9.Gq7kxBFtin905k5cmR_pPVp2QPGSd4uUeeX5DSqU58U"

FROM_DATE = '2025-03-01'
TO_DATE = '2025-03-30'

# Tata Motors instrument keys (using ISIN)
INSTRUMENT_KEYS = {
    'NSE': 'NSE_EQ|INE155A01022',  # Tata Motors NSE
    'BSE': 'BSE_EQ|INE155A01022'   # Tata Motors BSE
}

class TataMotorsDataFetcher:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {ACCESS_TOKEN}'
        }
        self.base_url = 'https://api.upstox.com/v2'

    def fetch_ohlc(self, instrument_key, exchange):
        url = f'{self.base_url}/historical-candle/{instrument_key}/1minute/{TO_DATE}/{FROM_DATE}'
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data['data']['candles']
        except requests.exceptions.HTTPError as err:
            print(f"\nError fetching {exchange} data:")
            print(f"URL: {url}")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        except Exception as err:
            print(f"Unexpected error fetching {exchange} data: {err}")
            return None

    def save_to_csv(self, df, exchange):
        if df is None or df.empty:
            print(f"No data to save for {exchange}")
            return None
        
        # Clean and format data
        df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%d-%m-%Y %H:%M')
        df = df.sort_values('Timestamp')
        
        filename = f"TATA_MOTORS_{exchange}_{FROM_DATE}_to_{TO_DATE}.csv"
        df.to_csv(filename, index=False)
        return filename

    def run(self):
        print(f"\nFetching Tata Motors data from {FROM_DATE} to {TO_DATE}")

        for exchange, instrument_key in INSTRUMENT_KEYS.items():
            print(f"\nProcessing {exchange} with key {instrument_key}...")
            data = self.fetch_ohlc(instrument_key, exchange)
            
            if data:
                df = pd.DataFrame(data)
                filename = self.save_to_csv(df, exchange)
                print(f"Successfully saved {exchange} data to {filename}")
            else:
                print(f"Failed to fetch {exchange} data")

if __name__ == "__main__":
    fetcher = TataMotorsDataFetcher()
    fetcher.run()