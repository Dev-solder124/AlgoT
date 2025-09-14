import requests
import pandas as pd

# Replace these with your actual credentials
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIzRENQWlUiLCJqdGkiOiI2N2I2Yjc4ZWVlMzYwZDZiNDU3MzZlNzEiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaWF0IjoxNzQwMDI3NzkwLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NDAwODg4MDB9.1kiPnfNWbuxFCgXzQPQGuMBX4juF43eOY-xtZyH6yiE"

# Instrument key for Nifty 50
INSTRUMENT_KEY = 'NSE_INDEX|Nifty Bank'

FROM_DATE='2025-02-07'
TO_DATE='2025-03-03'

# Upstox API endpoint for historical candles DATES FORMAT:/TODATE/FROMDAT{YYYY-MM-DD}
url = f'https://api.upstox.com/v2/historical-candle/{INSTRUMENT_KEY}/1minute/{TO_DATE}/{FROM_DATE}'

# Headers for the API request
headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {ACCESS_TOKEN}'
}

# Function to fetch OHLC data
def fetch_ohlc_data():
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        #print("API Response:", data)  # Debugging: Print the API response
        return data['data']['candles']
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
        print(f"Response: {response.text}")  # Print the response for debugging
        return None
    except Exception as err:
        print(f"Error: {err}")
        return None


def save_dataframe_to_csv(df, from_date, to_date, filename_prefix="nifty50_upx_"):
    """
    Saves the DataFrame to a CSV file with meaningful column names and a timestamp-based filename.
    The DataFrame is sorted so that the most recent data is at the bottom.

    Args:
        df (pd.DataFrame): The DataFrame containing OHLC data.
        from_date (str): Start date of the data in 'YYYY-MM-DD' format.
        to_date (str): End date of the data in 'YYYY-MM-DD' format.
        filename_prefix (str): Prefix for the CSV filename. Default is "nifty50_ohlc".
    """
    # Assign meaningful column names
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdditionalColumn']
    
    # Convert the Timestamp column to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Format the Timestamp column to 'DD-MM-YYYY HH:MM:SS'
    df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Sort the DataFrame by Timestamp in ascending order (oldest first)
    df = df.sort_values(by='Timestamp', ascending=True)
    
    # Generate the filename with from_date and to_date
    filename = f"BANK{filename_prefix}_{from_date}_to_{to_date}.csv"
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Main function
def main():
    # Fetch OHLC data
    ohlc_data = fetch_ohlc_data()
    if ohlc_data:
        # Convert to DataFrame
        df = pd.DataFrame(ohlc_data)
        save_dataframe_to_csv(df,FROM_DATE,TO_DATE)
        

    else:
        print("No data found for the specified date range.")

if __name__ == "__main__":
    main()