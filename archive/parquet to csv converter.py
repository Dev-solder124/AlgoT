import pandas as pd

# Read DataFrame from a .parquet file
input_file = "D:\AlgoT\data\stocks\RELIANCE_EQ.parquet"   # change to your parquet file path
df = pd.read_parquet(input_file)

# Save DataFrame to a .csv file
output_file = "D:\AlgoT\data\RELIANCE_EQ.csv"     # desired csv file path
df.to_csv(output_file, index=False)

print(f"Converted '{input_file}' to '{output_file}' successfully.")
