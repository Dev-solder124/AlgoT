import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def exponential_moving_average(data, window):
    alpha=2/(window+1)

    ema = [data[0]]  # Start with the first data point as the initial EMA
    for i in range(1, len(data)):
        ema_value = alpha * data[i] + (1 - alpha) * ema[i - 1]
        ema.append(ema_value)
    return ema




rown=[]


with open("d:\\AlgoT\\Book1.csv",'r') as f:
    fread=csv.reader(f)
    for i in fread:
        rown.append(i)

    print("total lines read",fread.line_num)

#[row[date0][open1][high2][low3][close4]]


row= np.array(rown).T.tolist()

datetime_strings=row[0][1:]
datetime_objects = [datetime.strptime(dt, '%d-%m-%Y %H:%M') for dt in datetime_strings]

open_values=row[1][1:]
open_values=[float(v) for v in open_values]

high_values=row[2][1:]
high_values=[float(v) for v in high_values]

low_values=row[3][1:]
low_values=[float(v) for v in low_values]

close_values=row[4][1:]
close_values=[float(v) for v in close_values]


ema_values=exponential_moving_average(close_values,75)






#plt.xkcd()
plt.figure(figsize=(10, 6))
#plt.plot(datetime_objects, close_values, linestyle='-')
plt.plot(datetime_objects, open_values, linestyle='-')
plt.plot(datetime_objects, ema_values, linestyle='-')
for i in range(1,len(datetime_objects)):
    plt.plot(datetime_objects[i],ema_values[i],marker='o',color='r')

#plt.ylim(min(values) - 1, max(values) + 1)  # Add some padding
# Formatting the plot
plt.title('Datetime vs Values')
plt.xlabel('Datetime')
plt.ylabel('Values')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout() 

plt.show()

