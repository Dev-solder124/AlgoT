import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


net=0
Ptrade=0
Ltrade=0
rown=[]
n=375
day=0

start_day=1+(day*n)


target_increment=3
stoploss_increment=1

net_data=[]
net_data_time=[]


Ppoints=0
Lpoints=0



print_processs=1

with open(r"D:\AlgoT\TATA_MOTORS_BSE_2025-03-01_to_2025-03-30.csv",'r') as f:
    fread=csv.reader(f)
    for i in fread:
        rown.append(i)

    print("total lines read",fread.line_num)

#[row[date0][open1][high2][low3][close4]]


row= np.array(rown).T.tolist()

datetime_strings=row[0][start_day:]
datetime_objects = [datetime.strptime(dt, '%d-%m-%Y %H:%M') for dt in datetime_strings]


close_values=row[4][start_day:]
close_values=[float(v) for v in close_values]

# ema_values=exponential_moving_average(close_values,75)


#plt.xkcd()
plt.figure(figsize=(10, 6))

plt.plot(datetime_objects, close_values,color='gray', linestyle='-')


net=0
Ptrade=0
Ltrade=0
rown=[]
n=375
day=0

start_day=1+(day*n)


target_increment=3
stoploss_increment=1

net_data=[]
net_data_time=[]


Ppoints=0
Lpoints=0



print_processs=1

with open(r"D:\AlgoT\TATA_MOTORS_NSE_2025-03-01_to_2025-03-30.csv",'r') as f:
    fread=csv.reader(f)
    for i in fread:
        rown.append(i)

    print("total lines read",fread.line_num)

#[row[date0][open1][high2][low3][close4]]


row= np.array(rown).T.tolist()

datetime_strings=row[0][start_day:]
datetime_objects = [datetime.strptime(dt, '%d-%m-%Y %H:%M') for dt in datetime_strings]


close_values=row[4][start_day:]
close_values=[float(v) for v in close_values]

# ema_values=exponential_moving_average(close_values,7

plt.plot(datetime_objects, close_values,color='b', linestyle='-')
plt.show()