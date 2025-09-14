'''
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

        if position=="Long":
            if bar_closevalue<=stoploss:
                position="Empty"
                plt.plot(plot_times,plot_bar,color='r')
                plot_bar.clear()
                plot_times.clear()

            elif bar_closevalue>=target:
                position="Empty"
                plt.plot(plot_times,plot_bar,color='g')
                plot_bar.clear()
                plot_times.clear()

            else:
                plot_bar.extend(bar_close)
                plot_times.extend(bar_time)


        elif position=="Short":
            if bar_closevalue>=stoploss:
                position="Empty"
                plt.plot(plot_times,plot_bar,color='r')
                plot_bar.clear()
                plot_times.clear()

            elif bar_closevalue<=target:
                position="Empty"
                plt.plot(plot_times,plot_bar,color='g')
                plot_bar.clear()
                plot_times.clear()


            else:
                plot_bar.extend(bar_close)
                plot_times.extend(bar_time)

        elif position=="Empty":
            if curr_low>ema_values[i]:
                position="Short"
                stoploss=curr_high
                target=bar_closevalue-2*(stoploss-bar_closevalue)
                
            elif curr_high<ema_values[i]:
                position="Long"
                stoploss=curr_low
                target=bar_closevalue+2*(bar_closevalue-stoploss)

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
'''