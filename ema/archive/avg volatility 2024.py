import pandas as pd

# Load the CSV file
file_path = r'D:\AlgoT\NIFTY 50-01-01-2024-to-01-01-2025.csv'
try:
    data = pd.read_csv(file_path)
    
    # Strip leading and trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Check if 'High' and 'Low' columns exist
    if 'High' in data.columns and 'Low' in data.columns:
        # Calculate the daily difference between High and Low
        data['Difference'] = data['High'] - data['Low']
        
        # Calculate average, lowest, and highest differences
        average_difference = data['Difference'].mean()
        lowest_difference = data['Difference'].min()
        highest_difference = data['Difference'].max()
        
        # Print results
        print(f"The average daily difference is: {average_difference:.2f}")
        print(f"The lowest daily difference is: {lowest_difference:.2f}")
        print(f"The highest daily difference is: {highest_difference:.2f}")
    else:
        print("The required columns ('High' and/or 'Low') are missing from the dataset.")
except FileNotFoundError:
    print(f"File not found. Please check the file path: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
