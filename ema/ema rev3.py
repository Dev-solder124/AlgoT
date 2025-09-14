import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from dateutil.parser import parse

net=0
Ptrade=0
Ltrade=0
rown=[]
n=375
day=0

start_day=1+(day*n)

target_increment=10
stoploss_increment=1

net_data=[]
net_data_time=[]

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

datetime_strings=row[0][start_day:]
datetime_objects = [datetime.strptime(dt, '%d-%m-%Y %H:%M') for dt in datetime_strings]
# datetime_objects = [parse(dt) for dt in datetime_strings]

open_values=row[1][start_day:]
open_values=[float(v) for v in open_values]

high_values=row[2][start_day:]
high_values=[float(v) for v in high_values]

low_values=row[3][start_day:]
low_values=[float(v) for v in low_values]

close_values=row[4][start_day:]
close_values=[float(v) for v in close_values]

# ema_values=exponential_moving_average(close_values,75)


#plt.xkcd()
plt.figure(figsize=(10, 6))
plt.plot(datetime_objects, close_values,color='gray', linestyle='-')
#plt.plot(datetime_objects, high_values,color='m', linestyle='-',linewidth=1)
#plt.plot(datetime_objects, low_values,color='y', linestyle='-',linewidth=1)
# plt.plot(datetime_objects, ema_values,linewidth=1, linestyle='-')

r=[]
bar15=0
bar5=0
bar15_high=[]
bar5_high=[]
bar15_low=[]
bar5_low=[]
bar15_open=[]
bar5_open=[]
bar15_close=[]
bar5_close=[]
bar15_time=[]
bar5_time=[]
bar15_closevalue=0
curr15_high=close_values[0]
curr15_low=close_values[0]
curr5_high=close_values[0]
curr5_low=close_values[0]
previous_high=0
previous_low=0
position="Empty"
seek_status="Empty"
entry_target=0 
plot_bar=[]
plot_times=[]
seek_bar=[]
seek_time=[]
temp_stoploss=0
stoploss=0
target=0
buffer=0
trtbuffer=0
entry=0
gap=0

real_ema=[]
real_ema_time=[]
previous_ema=close_values[0]

window=75
alpha=2/(window+1)


for i in range(len(datetime_objects)):
    
    ema_value = alpha * close_values[i] + (1 - alpha) * previous_ema
    previous_ema=ema_value

    real_ema.append(ema_value)
    real_ema_time.append(datetime_objects[i])

    if datetime_objects[i].hour==9 and datetime_objects[i].minute==15:
        if print_processs:
            print("-----------------------------------\n",datetime_strings[i],"\n-----------------------------------")


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
                print(datetime_strings[i],"Long exit at",close_values[i],"with",close_values[i]-entry,"points")
            net+=close_values[i]-entry



        elif close_values[i]>=target:
            position="Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])

            plt.plot(plot_times,plot_bar,color='g',marker='^')

            plot_bar.clear()
            plot_times.clear()
            if print_processs:
                print(datetime_strings[i],"Long exit at",close_values[i],"with",close_values[i]-entry,"points")
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
                print(datetime_strings[i],"Short exit at",close_values[i],"with",-close_values[i]+entry,"points")
            net+=-close_values[i]+entry
            


        elif close_values[i]<=target:
            position="Empty"
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i])

            plt.plot(plot_times,plot_bar,color='g',marker='v')

            plot_bar.clear()
            plot_times.clear()
            if print_processs:
                print(datetime_strings[i],"Short exit at",close_values[i],"with",-close_values[i]+entry,"points")
            net+=-close_values[i]+entry

            Ptrade+=1
            Ppoints+=-close_values[i]+entry


        else:
            plot_bar.append(close_values[i])
            plot_times.append(datetime_objects[i]) 


    if bar5==5:
        curr5_high=max(bar5_high)
        curr5_low=min(bar5_low)
        bar5_closevalue=bar5_close[-1]

        if seek_status=="Long seeking" and position=="Empty":
            seek_time.extend(bar5_time)
            seek_bar.extend(bar5_close)
            if close_values[i]>=entry_target:
                position="Long"
                seek_status="Empty"
                plot_bar.append(close_values[i])
                plot_times.append(datetime_objects[i])
                entry=close_values[i]
                
                buffer=abs(entry_target-temp_stoploss)
                trtbuffer=buffer
                if buffer==0:
                    buffer=0.5       
                if buffer>30:
                    buffer=30
                stoploss=entry-(stoploss_increment*buffer)
                target=entry+target_increment*trtbuffer
                if print_processs:
                    print(datetime_strings[i],position,"entry=",entry,"buffer:",buffer,"SL",stoploss,"Tgrt",target)


                plt.plot(seek_time,seek_bar,color='pink')  
                seek_bar=[]
                seek_time=[]        


            elif close_values[i]<=temp_stoploss:
                if print_processs:
                        print("Long abandoned, Short taken")
                position="Short"
                seek_status="Empty"
                plot_bar.append(close_values[i])
                plot_times.append(datetime_objects[i])
                entry=close_values[i]
                buffer=abs(-temp_stoploss+entry_target)
                trtbuffer=buffer
                if buffer==0:
                    buffer=0.5
                if buffer>30:
                    buffer=30   
                target=entry-target_increment*trtbuffer
                stoploss=entry+stoploss_increment*buffer
                if print_processs:
                    print(datetime_strings[i],position,"entry=",entry,"buffer:",buffer,"SL",stoploss,"Tgrt",target)


                plt.plot(seek_time,seek_bar,color='pink')  
                seek_bar=[]
                seek_time=[] 
            
            


        if seek_status=="Short seeking" and position=="Empty":
            seek_time.extend(bar5_time)
            seek_bar.extend(bar5_close)
            if close_values[i]<=entry_target:
                position="Short"
                seek_status="Empty"
                plot_bar.append(close_values[i])
                plot_times.append(datetime_objects[i])
                entry=close_values[i]
                stoploss=temp_stoploss#curr5_high
                buffer=abs(temp_stoploss-entry_target)
                trtbuffer=buffer
                if buffer==0:
                    buffer=0.5
                if buffer>30:
                    buffer=30  
                target=entry-target_increment*trtbuffer
                stoploss=entry+stoploss_increment*buffer

                if print_processs:
                    print(datetime_strings[i],position,"entry=",entry,"buffer:",buffer,"SL",stoploss,"Tgrt",target)

                plt.plot(seek_time,seek_bar,color='pink')  
                seek_bar=[]
                seek_time=[]        

            elif close_values[i]>=temp_stoploss:
                
                if print_processs:
                        print("Short abandoned, Long taken")
                position="Long"
                seek_status="Empty"
                plot_bar.append(close_values[i])
                plot_times.append(datetime_objects[i])
                entry=close_values[i]
                buffer=abs(temp_stoploss-entry_target)
                trtbuffer=buffer
                stoploss=temp_stoploss
                if buffer==0:
                    buffer=0.5 
                if buffer>30:
                    buffer=30             
                target=entry+target_increment*trtbuffer 
                stoploss=entry-stoploss_increment*buffer 

                if print_processs:
                    print(datetime_strings[i],position,"entry=",entry,"buffer:",buffer,"SL",stoploss,"Tgrt",target)


                plt.plot(seek_time,seek_bar,color='pink')  
                seek_bar=[]
                seek_time=[]  



        bar5_high.clear()
        bar5_low.clear()
        bar5_open.clear()
        bar5_close.clear()
        bar5_time.clear()


        bar5=0
        
        bar5_high.append(high_values[i])
        bar5_low.append(low_values[i])
        bar5_open.append(open_values[i])
        bar5_close.append(close_values[i])
        bar5_time.append(datetime_objects[i])
        bar5+=1
    else:
        bar5_high.append(high_values[i])
        bar5_low.append(low_values[i])
        bar5_open.append(open_values[i])
        bar5_close.append(close_values[i])
        bar5_time.append(datetime_objects[i])
        bar5+=1
        

    if bar15==15:
        curr15_high=max(bar15_high)
        curr15_low=min(bar15_low)
        bar15_closevalue=bar15_close[-1]



        if seek_status=="Empty":
            if curr15_low>ema_value:
                seek_status="Short seeking"
                entry_target=curr15_low
                gap=curr15_low-ema_value
                temp_stoploss=curr15_high

                seek_bar.clear()
                seek_time.clear()
                seek_bar.append(close_values[i])
                seek_time.append(datetime_objects[i])

                if print_processs:
                    print(datetime_strings[i],seek_status)

                
            elif curr15_high<ema_value:
                seek_status="Long seeking"
                entry_target=curr15_high
                gap=ema_value-curr15_high
                temp_stoploss=curr15_low

                
                seek_bar.clear()
                seek_time.clear()

                seek_bar.append(close_values[i])
                seek_time.append(datetime_objects[i])

                if print_processs:
                    print(datetime_strings[i],seek_status)

        bar15_high.clear()
        bar15_low.clear()
        bar15_open.clear()
        bar15_close.clear()
        bar15_time.clear()

        bar15=0
        
        bar15_high.append(high_values[i])
        bar15_low.append(low_values[i])
        bar15_open.append(open_values[i])
        bar15_close.append(close_values[i])
        bar15_time.append(datetime_objects[i])
        bar15+=1    
    else:
        bar15_high.append(high_values[i])
        bar15_low.append(low_values[i])
        bar15_open.append(open_values[i])
        bar15_close.append(close_values[i])
        bar15_time.append(datetime_objects[i])
        bar15+=1
    net_data.append(net)
    net_data_time.append(datetime_objects[i])

    if datetime_objects[i].hour==15 and datetime_objects[i].minute==29:
        seek_bar=[]
        seek_time=[]
        if seek_status=="Long seeking" or seek_status=="Short seeking":
            seek_status="Empty"
        bar15_high.clear()
        bar15_low.clear()
        bar15_open.clear()
        bar15_close.clear()
        bar15_time.clear()

        bar15=0

        bar5_high.clear()
        bar5_low.clear()
        bar5_open.clear()
        bar5_close.clear()
        bar5_time.clear()

        bar5=0

plt.plot(net_data_time,net_data,color='y',linestyle='-')
# plt.plot(real_ema_time,real_ema,color='b',linestyle='-')
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
print("Avg points per trade:", net/(Ptrade+Ltrade))
print("Cost per trade: 1  ")
print("Total no of trades:",Ptrade+Ltrade)
if Ltrade!=0:
    # print(f"P:L= {Ptrade/Ltrade}")
    print(f"P%= {(Ptrade/(Ptrade+Ltrade))*100}")

iso_timestamps = []
for ts in datetime_objects:
    dt = ts
    iso_timestamps.append(dt.isoformat(timespec='seconds'))
print("Average Risk to Reward Ratio 1:",(Ppoints/Ptrade)/(abs(Lpoints)/Ltrade))
np.save('net.npy', np.array(net_data))
np.save('date.npy', np.array(iso_timestamps))
plt.show()