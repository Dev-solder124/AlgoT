import pandas as pd

df=pd.read_csv(r"D:\AlgoT\trades.csv")
df = df.dropna(how='all')
# df2=pd.read_csv(r"D:\AlgoT\ML F NIFTY100.csv")
# df3=pd.read_csv(r"D:\AlgoT\ML F NIFTY200.csv")
# df4=pd.read_csv(r"D:\AlgoT\ML F NIFTY500.csv")
# df5=pd.read_csv(r"D:\AlgoT\ML F NIFTYAUTO.csv")
# df6=pd.read_csv(r"D:\AlgoT\ML F NIFTYCOMMOD.csv")

# cdf = pd.concat([df2, df3], ignore_index=True)
print(df)

df=df.drop(columns=['POSITION'])
# df.to_csv("trades2.csv",index=False)
print(df)
print(df.info())