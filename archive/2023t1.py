import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

rown=[]

n=200


with open("d:\\AlgoT\\Book1.csv",'r') as f:
    fread=csv.reader(f)
    for i in fread:
        rown.append(i)

    print("total lines read",fread.line_num)

#[row[date0][open1][high2][low3][close4]]


row= np.array(rown).T.tolist()

datetime_strings=row[0][1:(n+1)]
datetime_objects = [datetime.strptime(dt, '%d-%m-%Y %H:%M') for dt in datetime_strings]

open_values=row[1][1:(n+1)]
open_values=[float(v) for v in open_values]

high_values=row[2][1:(n+1)]
high_values=[float(v) for v in high_values]

low_values=row[3][1:(n+1)]
low_values=[float(v) for v in low_values]

close_values=row[4][1:(n+1)]
close_values=[float(v) for v in close_values]


#plt.xkcd()
plt.figure(figsize=(10, 6))
#plt.plot(datetime_objects, close_values, linestyle='-')
#plt.plot(datetime_objects, open_values, linestyle='-')

r=[]
bar=0
bar_high=[]
bar_low=[]
bar_open=[]
bar_close=[]
bar_time=[]
bar_closevalue=0
curr_high=close_values[0]
curr_low=close_values[0]
previous_high=0
previous_low=0
position="Sell"
plot_bar=[]
plot_times=[]
for i in range(0,n):
    if bar==15:
        previous_high=curr_high
        previous_low=curr_low

        curr_high=max(bar_high)
        curr_low=min(bar_low)
        bar_closevalue=bar_close[-1]
        if(position=="Buy"):
            print(position,bar_closevalue,previous_low,bar_close[0],bar_time[-1])

            plot_bar.extend(bar_close)
            plot_times.extend(bar_time)
            if(bar_closevalue<previous_low):
                if(bar_closevalue>=bar_close[0]):
                    plt.plot(plot_times, plot_bar,color='g', linestyle='-')
                else:
                    plt.plot(plot_times, plot_bar,color='r', linestyle='-')
                plot_bar.clear()
                plot_times.clear()

                position="Sell"

        else:
            print(position,bar_closevalue,previous_high,bar_close[0])

            plot_bar.extend(bar_close)
            plot_times.extend(bar_time)

            if(bar_closevalue>previous_high):
                position="Buy"
                if(bar_closevalue<=bar_close[0]):
                    plt.plot(plot_times, plot_bar,color='g', linestyle='-')
                else:
                    plt.plot(plot_times, plot_bar,color='r', linestyle='-')
                plot_bar.clear()
                plot_times.clear()
                
        bar_high.clear()
        bar_low.clear()
        bar_open.clear()
        bar_close.clear()
        bar_time.clear()


        bar=0
        
        bar_high.append(high_values[i])
        bar_low.append(low_values[i])
        bar_open.append(open_values[i])
        bar_close.append(close_values[i])
        bar_time.append(datetime_objects[i])
        bar+=1

    else:
        bar_high.append(high_values[i])
        bar_low.append(low_values[i])
        bar_open.append(open_values[i])
        bar_close.append(close_values[i])
        bar_time.append(datetime_objects[i])
        bar+=1
plt.plot(datetime_objects,low_values, linestyle='-')
plt.title('Datetime vs Values')
plt.xlabel('Datetime')
plt.ylabel('Values')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout() 

plt.show()

