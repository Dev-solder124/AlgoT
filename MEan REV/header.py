import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

n=375
day=0

start_day=1+(day*n)

diff_data=[]
diff_data_time=[]

BSE=[]
NSE=[]

with open(r"D:\AlgoT\TATA_MOTORS_BSE_2025-03-01_to_2025-03-30.csv",'r') as f:
    fread=csv.reader(f)
    for i in fread:
        BSE.append(i)

    print("total lines read bse",fread.line_num)

with open(r"D:\AlgoT\TATA_MOTORS_NSE_2025-03-01_to_2025-03-30.csv",'r') as f:
    fread=csv.reader(f)
    for i in fread:
        NSE.append(i)

    print("total lines read nse",fread.line_num)


Ndat= np.array(NSE).T.tolist()
Bdat= np.array(BSE).T.tolist()


Ndatetime_strings=Ndat[0][start_day:]
Ndatetime_objects = [datetime.strptime(dt, '%d-%m-%Y %H:%M') for dt in Ndatetime_strings]

Nclose_values=Ndat[4][start_day:]
Nclose_values=[float(v) for v in Nclose_values]

Bdatetime_strings=Bdat[0][start_day:]
Bdatetime_objects = [datetime.strptime(dt, '%d-%m-%Y %H:%M') for dt in Bdatetime_strings]

Bclose_values=Bdat[4][start_day:]
Bclose_values=[float(v) for v in Bclose_values]

plt.figure(figsize=(10, 6))
plt.plot(Ndatetime_objects, Nclose_values,color='purple', linestyle='-')
plt.plot(Bdatetime_objects, Bclose_values,color='pink', linestyle='-')


for i in range(0,len(Bdatetime_objects)):
    diff_data.append((Nclose_values[i]-Bclose_values[i]))
    diff_data_time.append(Ndatetime_objects[i])

plt.plot(diff_data_time,diff_data,color='olive',linestyle='-')
plt.title('Datetime vs Values')
plt.xlabel('Datetime')
plt.ylabel('Values')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout() 
plt.show()

    
