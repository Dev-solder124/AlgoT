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

def moving_average_all_elements(data, window_size):
    if not data:
        return []

    sma_values = []

    for i in range(len(data)):
        # Adjust the window size for beginning elements
        current_window_size = min(i + 1, window_size)
        window = data[max(0, i - window_size + 1) : i + 1]  # Dynamic window selection
        sma = sum(window) / current_window_size
        sma_values.append(sma)

    return sma_values

net=0
Ptrade=0
Ltrade=0
rown=[]
n=375
day=20
start_day=1#+(day*n)
end_day=n*(day+1)

target_increment=10
stoploss_increment=1



Ppoints=0
Lpoints=0



print_processs=0

with open(r"D:\AlgoT\nifty50_upx__2025-01-01_to_2025-01-30.csv",'r') as f:
    fread=csv.reader(f)
    for i in fread:
        rown.append(i)

    print("total lines read",fread.line_num)

#[row[date0][open1][high2][low3][close4]]


row= np.array(rown).T.tolist()

datetime_strings=row[0][start_day:(end_day+1)]
datetime_objects = [datetime.strptime(dt, '%d-%m-%Y %H:%M') for dt in datetime_strings]

open_values=row[1][start_day:(end_day+1)]
open_values=[float(v) for v in open_values]

high_values=row[2][start_day:(end_day+1)]
high_values=[float(v) for v in high_values]

low_values=row[3][start_day:(end_day+1)]
low_values=[float(v) for v in low_values]

close_values=row[4][start_day:(end_day+1)]
close_values=[float(v) for v in close_values]

ema_values=exponential_moving_average(close_values,10)

ema_values2=exponential_moving_average(close_values,50)

movin_avg=moving_average_all_elements(close_values,75)

movin_avg2=moving_average_all_elements(close_values,150)


#plt.xkcd()
plt.figure(figsize=(10, 6))
plt.plot(datetime_objects, close_values,color='gray', linestyle='-')
#plt.plot(datetime_objects, high_values,color='m', linestyle='-',linewidth=1)
#plt.plot(datetime_objects, low_values,color='y', linestyle='-',linewidth=1)
plt.plot(datetime_objects, ema_values,linewidth=1,color='m', linestyle='-')
plt.plot(datetime_objects, ema_values2,linewidth=1,color='g', linestyle='-')
#plt.plot(datetime_objects, movin_avg,linewidth=1, linestyle='-')
#plt.plot(datetime_objects, movin_avg2,linewidth=1, linestyle='-')


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
entry_target=0
plot_bar=[]
plot_times=[]
stoploss=0
target=0
buffer=0
entry=0
gap=0
for i in range(0,end_day-start_day):
    
    if bar==15:
        curr_high=max(bar_high)
        curr_low=min(bar_low)
        bar_closevalue=bar_close[-1]

        #if position=="Long seeking":
        #if(curr_high>ema_values[i] and curr_low>ema_values[i]):
            #plt.plot(datetime_objects[i],close_values[i],marker='o')
        
        #if(curr_high<ema_values[i] and curr_low<ema_values[i]):
            #plt.plot(datetime_objects[i],close_values[i],marker='o')





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

   
plt.title('Datetime vs Values')
plt.xlabel('Datetime')
plt.ylabel('Values')
plt.grid(True)
plt.xticks(rotation=10)  # Rotate x-axis labels for better readability
plt.tight_layout() 



print("Net P/L:",net)
print("Ppoints",Ppoints)
print("Lpoints",Lpoints)
print("Ptrades:",Ptrade)
print("Ltrades:",Ltrade)
print("Total no of trades:",Ptrade+Ltrade)
if Ltrade!=0:
    print(f"P:L= {Ptrade/Ltrade}")
    print(f"P%= {(Ptrade/(Ptrade+Ltrade))*100}")


plt.show()