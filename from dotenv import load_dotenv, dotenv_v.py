import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
from local_extreme import LocalExtreme, extremes_sanity_checks


rown=[]
n=375
day=0

start_day=1+(day*n)

print_processs=0

with open(r"D:\AlgoT\nifty50_upx__2025-04-21_to_2025-04-30.csv",'r') as f:
    fread=csv.reader(f)
    for i in fread:
        rown.append(i)

    print("total lines read",fread.line_num)

#[row[date0][open1][high2][low3][close4]]


row= np.array(rown).T.tolist()

datetime_strings=row[0][start_day:]
datetime_objects = [datetime.strptime(dt, '%d-%m-%Y %H:%M') for dt in datetime_strings]

open_values=row[1][start_day:]
open_values=[float(v) for v in open_values]

high_values=row[2][start_day:]
high_values=[float(v) for v in high_values]

low_values=row[3][start_day:]
low_values=[float(v) for v in low_values]

close_values=row[4][start_day:]
close_values=[float(v) for v in close_values]

drhighvalue=[]
drtime=[]

drlowvalue=[]

drvalue=[]

plt.figure(figsize=(10, 6))
plt.plot(datetime_objects, close_values,color='gray', linestyle='-')

    
for i in range(1,len(close_values)-1):
    if (close_values[i]>=close_values[i+1] and close_values[i]>=close_values[i-1]):

        drhighvalue.append(close_values[i])
        drvalue.append(close_values[i])
        drtime.append(datetime_objects[i])

    if (close_values[i]<=close_values[i+1] and close_values[i]<=close_values[i-1]):
        drlowvalue.append(close_values[i])
        drvalue.append(close_values[i])
        drtime.append(datetime_objects[i])

dr2=[]
dr2time=[]
dr2high=[]
dr2low=[]

lastindx=0
for  i in range(2,len(drvalue)-2):
    if drvalue[i] in drhighvalue:
        if( drvalue[i]>drvalue[i-2] and drvalue[i]>drvalue[i+2]):
            if dr2!=[] and (dr2[-1] in drhighvalue):
                si=lastindx
                ei=i
                # plt.plot(drtime[si:ei],drvalue[si:ei],color='pink',linestyle='-')
                if si!=ei:
                    lp=min(drvalue[si:ei])
                    li=drvalue.index(lp,si,ei)
                    dr2.append(lp)
                    dr2low.append(lp)
                    dr2time.append(drtime[li])      
            dr2.append(drvalue[i])
            dr2high.append(drvalue[i])
            dr2time.append(drtime[i])
            lastindx=i
            # plt.plot(drtime[i],drvalue[i],marker='^',color='g')



    elif drvalue[i] in drlowvalue:
        if( drvalue[i]<drvalue[i-2] and drvalue[i]<drvalue[i+2]):
            if dr2!=[] and (dr2[-1] in drlowvalue):
                si=lastindx
                ei=i
                # plt.plot(drtime[si:ei],drvalue[si:ei],color='pink',linestyle='-')
                if si!=ei:
                    lp=max(drvalue[si:ei])
                    li=drvalue.index(lp,si,ei)
                    dr2.append(lp)
                    dr2high.append(lp)
                    dr2time.append(drtime[li])
            dr2.append(drvalue[i])
            dr2low.append(drvalue[i])
            dr2time.append(drtime[i])
            lastindx=i
            # plt.plot(drtime[i],drvalue[i],marker='v',color='r')
            

 
dr3=[]
dr3time=[]           
for  i in range(1,len(dr2)-1):
    if (dr2[i]>dr2[i+1] and dr2[i]>dr2[i-1]):
        # drhighvalue.append(dr2[i])
        dr3.append(dr2[i])
        dr3time.append(dr2time[i])

    if (dr2[i]<dr2[i+1] and dr2[i]<dr2[i-1]):
        # drlowvalue.append(dr2[i])
        dr3.append(dr2[i])
        dr3time.append(dr2time[i])






plt.plot(drtime,drvalue,color='g',linestyle='-')
plt.plot(dr2time,dr2,color='b',linestyle='-')
# plt.plot(dr3time,dr3,color='g',linestyle='-')

plt.show()

