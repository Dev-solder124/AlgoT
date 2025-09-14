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
net=0
Ptrade=0
Ltrade=0
rown=[]
n=375
day=2200
start_day=1#+(day*n)
end_day=n*(day+1)

target_increment=10
stoploss_increment=1



Ppoints=0
Lpoints=0



print_processs=0

with open(r"D:\AlgoT\NIFTY 50 COPY.csv",'r') as f:
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

ema_values=exponential_moving_average(close_values,75)


#plt.xkcd()
plt.figure(figsize=(10, 6))
plt.plot(datetime_objects, close_values,color='gray', linestyle='-')
#plt.plot(datetime_objects, high_values,color='m', linestyle='-',linewidth=1)
#plt.plot(datetime_objects, low_values,color='y', linestyle='-',linewidth=1)
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
entry_target=0
plot_bar=[]
plot_times=[]
stoploss=0
target=0
buffer=0
entry=0
gap=0
for i in range(0,end_day-start_day):
    if position=="Long":
        if ((close_values[i]<=stoploss) or (datetime_objects[i].hour==15 and datetime_objects[i].minute==29)):
            position="Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            if(close_values[i]-entry<=0):
                plt.plot(plot_times,plot_bar,color='r',marker='^')
                Ltrade+=1
                Lpoints+=close_values[i]-entry
            else:
                plt.plot(plot_times,plot_bar,color='g',marker='^')
                Ptrade+=1
                Ppoints+=close_values[i]-entry
            plot_bar.clear()
            plot_times.clear()
            if print_processs:
                print("Long exit at",close_values[i],"with",close_values[i]-entry,"points")
            net+=close_values[i]-entry



        elif close_values[i]>=target:
            position="Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])

            plt.plot(plot_times,plot_bar,color='g',marker='^')

            plot_bar.clear()
            plot_times.clear()
            if print_processs:
                print("Long exit at",close_values[i],"with",close_values[i]-entry,"points")
            net+=close_values[i]-entry

            Ptrade+=1
            Ppoints+=close_values[i]-entry


        else:
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])




    if position=="Short":
        if ((close_values[i]>=stoploss) or (datetime_objects[i].hour==15 and datetime_objects[i].minute==29)):
            position="Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])
            if(-close_values[i]+entry<=0):
                plt.plot(plot_times,plot_bar,color='r',marker='v')
                Ltrade+=1
                Lpoints+=-close_values[i]+entry
            else:
                plt.plot(plot_times,plot_bar,color='g',marker='v')
                Ptrade+=1
                Ppoints+=-close_values[i]+entry
            plot_bar.clear()
            plot_times.clear()
            if print_processs:
                print("Long exit at",close_values[i],"with",-close_values[i]+entry,"points")
            net+=-close_values[i]+entry
            


        elif close_values[i]<=target:
            position="Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])

            plt.plot(plot_times,plot_bar,color='g',marker='v')

            plot_bar.clear()
            plot_times.clear()
            if print_processs:
                print("Long exit at",close_values[i],"with",-close_values[i]+entry,"points")
            net+=-close_values[i]+entry

            Ptrade+=1
            Ppoints+=-close_values[i]+entry


        else:
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i]) 

        

    if bar==15:
        curr_high=max(bar_high)
        curr_low=min(bar_low)
        bar_closevalue=bar_close[-1]

        #if position=="Long seeking":
            

                
        if position=="Long seeking":
            if close_values[i]>=entry_target:
                position="Long"
                plot_bar.append(close_values[i])
                plot_times.append(datetime_objects[i])
                entry=close_values[i]
                buffer=abs(close_values[i]-stoploss)
                if buffer==0:
                    buffer=0.5            
                target=entry+target_increment*buffer

            elif close_values[i]<=stoploss:
                position="Empty"
                if print_processs:
                        print("Long abandoned")      

        if position=="Short seeking":
            if close_values[i]<=entry_target:
                position="Short"
                plot_bar.append(close_values[i])
                plot_times.append(datetime_objects[i])
                entry=close_values[i]
                buffer=abs(stoploss-close_values[i])
                if buffer==0:
                    buffer=0.5
                target=entry-target_increment*buffer

            elif close_values[i]>=stoploss:
                position="Empty"
                if print_processs:
                        print("Short abandoned")             


        if position=="Empty":
            if curr_low>ema_values[i]:
                position="Short seeking"
                entry_target=curr_low
                gap=curr_low-ema_values[i]
                stoploss=curr_high

                if print_processs:
                    print(position,"entry=",entry,"buffer:",buffer,"SL",stoploss,"Tgrt",target, "GAP:",gap)

                
            elif curr_high<ema_values[i]:
                position="Long seeking"
                entry_target=curr_high
                gap=ema_values[i]-curr_high
                stoploss=curr_low
                if print_processs:
                    print(position,"entry=",entry,"buffer:",buffer,"SL",stoploss,"Tgrt",target, "GAP:",gap)

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
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
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