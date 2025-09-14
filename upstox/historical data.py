import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import calendar
import time

load_dotenv("upstox.env")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

# Instrument key for Tata Motors
INSTRUMENT_KEY = "NSE_EQ|INE154A01025"  # tatamotors 'NSE_INDEX|Nifty 50'
name_prefix="ITC_EQ"
# Date range - can span multiple months
FROM_DATE = '2022-01-01'
TO_DATE = '2025-08-31'

def get_month_date_ranges(start_date_str, end_date_str):
    """
    Generate month-wise date ranges between start_date and end_date.
    
    Args:
        start_date_str (str): Start date in 'YYYY-MM-DD' format
        end_date_str (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        list: List of tuples containing (from_date, to_date) for each month
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    date_ranges = []
    current_date = start_date
    
    while current_date <= end_date:
        # Get the first day of current month
        month_start = current_date.replace(day=1)
        
        # Get the last day of current month
        last_day = calendar.monthrange(current_date.year, current_date.month)[1]
        month_end = current_date.replace(day=last_day)
        
        # Adjust the range boundaries
        range_start = max(month_start, start_date)
        range_end = min(month_end, end_date)
        
        date_ranges.append((
            range_start.strftime('%Y-%m-%d'),
            range_end.strftime('%Y-%m-%d')
        ))
        
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1, day=1)
    
    return date_ranges

def fetch_ohlc_data(from_date, to_date, retry_count=3, delay=1):
    """
    Fetch OHLC data for a specific date range with retry mechanism.
    
    Args:
        from_date (str): Start date in 'YYYY-MM-DD' format
        to_date (str): End date in 'YYYY-MM-DD' format
        retry_count (int): Number of retry attempts
        delay (int): Delay between retries in seconds
    
    Returns:
        list or None: List of candle data or None if failed
    """
    url = f'https://api.upstox.com/v2/historical-candle/{INSTRUMENT_KEY}/1minute/{to_date}/{from_date}'
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }
    
    for attempt in range(retry_count):
        try:
            print(f"Fetching data from {from_date} to {to_date} (Attempt {attempt + 1})")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if data exists
            if 'data' in data and 'candles' in data['data'] and data['data']['candles']:
                print(f"Successfully fetched {len(data['data']['candles'])} records")
                return data['data']['candles']
            else:
                print(f"No data available for {from_date} to {to_date}")
                return []
                
        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error (Attempt {attempt + 1}): {err}")
            if response.status_code == 429:  # Rate limit
                print(f"Rate limited. Waiting {delay * (attempt + 1)} seconds...")
                time.sleep(delay * (attempt + 1))
            elif attempt == retry_count - 1:
                print(f"Final attempt failed. Response: {response.text}")
                return None
        except Exception as err:
            print(f"Error (Attempt {attempt + 1}): {err}")
            if attempt == retry_count - 1:
                return None
        
        if attempt < retry_count - 1:
            time.sleep(delay)
    
    return None

def process_dataframe(df):
    """
    Process the DataFrame with proper column names and formatting.
    
    Args:
        df (pd.DataFrame): Raw DataFrame from API response
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    if df.empty:
        return df
    
    # Assign meaningful column names
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'AdditionalColumn']
    
    # Convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date to ensure proper order
    df = df.sort_values(by='date', ascending=True)
    
    # Remove duplicates if any
    df = df.drop_duplicates(subset=['date'], keep='first')
    
    return df

def save_dataframe_to_csv(df, from_date, to_date, filename_prefix=name_prefix):
    """
    Saves the DataFrame to a CSV file with meaningful column names and a date-based filename.
    
    Args:
        df (pd.DataFrame): The DataFrame containing OHLC data.
        from_date (str): Start date of the data in 'YYYY-MM-DD' format.
        to_date (str): End date of the data in 'YYYY-MM-DD' format.
        filename_prefix (str): Prefix for the CSV filename.
    """
    if df.empty:
        print("No data to save.")
        return
    
    # Format the date column to 'DD-MM-YYYY HH:MM'
    df_copy = df.copy()
    df_copy['date'] = df_copy['date'].dt.strftime('%d-%m-%Y %H:%M')
    
    # Generate the filename with from_date and to_date
    filename = f"{filename_prefix}_{from_date}_to_{to_date}.csv"
    
    # Save the DataFrame to a CSV file
    df_copy.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    print(f"Total records: {len(df_copy)}")
    print(f"Date range in data: {df['date'].min()} to {df['date'].max()}")

def fetch_multi_month_data(start_date, end_date):
    """
    Main function to fetch data across multiple months.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: Concatenated DataFrame with all data
    """
    print(f"Fetching data from {start_date} to {end_date}")
    
    # Get month-wise date ranges
    date_ranges = get_month_date_ranges(start_date, end_date)
    print(f"Data will be fetched in {len(date_ranges)} chunks:")
    for i, (from_dt, to_dt) in enumerate(date_ranges, 1):
        print(f"  {i}. {from_dt} to {to_dt}")
    
    all_dataframes = []
    
    for i, (from_date, to_date) in enumerate(date_ranges, 1):
        print(f"\n--- Chunk {i}/{len(date_ranges)} ---")
        
        # Fetch data for this month
        ohlc_data = fetch_ohlc_data(from_date, to_date)
        
        if ohlc_data is not None and len(ohlc_data) > 0:
            # Convert to DataFrame and process
            df = pd.DataFrame(ohlc_data)
            df_processed = process_dataframe(df)
            all_dataframes.append(df_processed)
            print(f"Added {len(df_processed)} records from {from_date} to {to_date}")
        else:
            print(f"No data retrieved for {from_date} to {to_date}")
        
        # Add delay between requests to avoid rate limiting
        if i < len(date_ranges):
            print("Waiting 2 seconds before next request...")
            time.sleep(2)
    
    # Concatenate all DataFrames
    if all_dataframes:
        print(f"\n--- Combining Data ---")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Final sort and cleanup
        combined_df = combined_df.sort_values(by='date', ascending=True)
        combined_df = combined_df.drop_duplicates(subset=['date'], keep='first')
        
        print(f"Total records after combining: {len(combined_df)}")
        return combined_df
    else:
        print("No data was retrieved for any date range.")
        return pd.DataFrame()

def main():
    """Main execution function"""
    print("=== Multi-Month OHLC Data Fetcher ===\n")
    
    # Validate date format
    try:
        datetime.strptime(FROM_DATE, '%Y-%m-%d')
        datetime.strptime(TO_DATE, '%Y-%m-%d')
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD format.")
        return
    
    # Check if date range is valid
    if FROM_DATE > TO_DATE:
        print("Error: FROM_DATE cannot be later than TO_DATE.")
        return
    
    # Fetch multi-month data
    final_df = fetch_multi_month_data(FROM_DATE, TO_DATE)
    
    # Save to CSV if data exists
    if not final_df.empty:
        save_dataframe_to_csv(final_df, FROM_DATE, TO_DATE)
        print(f"\n=== Summary ===")
        print(f"Successfully fetched and saved data from {FROM_DATE} to {TO_DATE}")
        print(f"Total records: {len(final_df)}")
        print(f"Data spans: {final_df['date'].min()} to {final_df['date'].max()}")
    else:
        print("\n=== Summary ===")
        print("No data was retrieved for the specified date range.")

if __name__ == "__main__":
    main()