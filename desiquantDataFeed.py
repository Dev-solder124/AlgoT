import s3fs
import pandas as pd

uri = "s3://desiquant/data/candles/SUNPHARMA/EQ.parquet.gz"

s3_params = {
"endpoint_url": "https://cbabd13f6c54798a9ec05df5b8070a6e.r2.cloudflarestorage.com",
"key": "5c8ea9c516abfc78987bc98c70d2868a", # FREE credentials for public access!
"secret": "0cf64f9f0b64f6008cf5efe1529c6772daa7d7d0822f5db42a7c6a1e41b3cadf", # FREE credentials for public access!
"client_kwargs": {
    "region_name": "auto"
    },
}

df = pd.read_parquet(uri, storage_options=s3_params)
df.head(10) # view sample data


# Save DataFrame to a .csv file
output_file = r"D:\AlgoT\data\SUNPHARMA.csv"     # desired csv file path
df.to_csv(output_file, index=False)