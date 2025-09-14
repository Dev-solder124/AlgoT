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

n=375


with open("d:\\AlgoT\\NIFTY 50.csv",'r') as f:
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

ema_values=exponential_moving_average(close_values,75)


#plt.xkcd()
plt.figure(figsize=(10, 6))
plt.plot(datetime_objects, close_values,color='gray', linestyle='-')
plt.plot(datetime_objects, ema_values,linewidth=1, linestyle='-')

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
position="Empty"
plot_bar=[]
plot_times=[]
stoploss=0
target=0
for i in range(0,n):
    if bar==15:
        curr_high=max(bar_high)
        curr_low=min(bar_low)
        bar_closevalue=bar_close[-1]

        if position=="Empty":
            if curr_low>ema_values[i]:
                position="Short"
                stoploss=curr_high
                target=bar_closevalue-3*(stoploss-bar_closevalue)
                
            elif curr_high<ema_values[i]:
                position="Long"
                stoploss=curr_low
                target=bar_closevalue+3*(bar_closevalue-stoploss)

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
    
    if position=="Long":
            if close_values[i]<=stoploss:
                position="Empty"
                plt.plot(plot_times,plot_bar,color='r')
                plot_bar.clear()
                plot_times.clear()

            elif close_values[i]>=target:
                position="Empty"
                plt.plot(plot_times,plot_bar,color='g')
                plot_bar.clear()
                plot_times.clear()

            else:
                plot_bar.append(close_values[i])
                plot_times.append(datetime_objects[i])


    elif position=="Short":
        if close_values[i]>=stoploss:
            position="Empty"
            plt.plot(plot_times,plot_bar,color='r')
            plot_bar.clear()
            plot_times.clear()

        elif close_values[i]<=target:
            position="Empty"
            plt.plot(plot_times,plot_bar,color='g')
            plot_bar.clear()
            plot_times.clear()


        else:
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])





plt.title('Datetime vs Values')
plt.xlabel('Datetime')
plt.ylabel('Values')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout() 

plt.show()

