import os
import pandas as pd
import talib as ta
import numpy as np
from tqdm import tqdm

origion = pd.read_csv('fu88.csv')
origion['datetime'] = pd.to_datetime(origion['datetime'])
origion.set_index('datetime',inplace=True)
origion = origion.dropna(axis=0)
origion.head()

data = origion.resample('1min').agg(
    {
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last'
    }
).dropna()

data['sma'] = ta.SMA(data['close'],3)
data['slope'] = ta.LINEARREG_SLOPE(data['sma'],3)

h1 = data[(data['slope'] >= 0.5)].index
l1 =  data[(data['slope'] <= -0.5)].index

data['sign']=0
data['sign'][h1] = 1
data['sign'][l1] = -1

test_data = data[-130000:-20000]

pos = 0

long_benifit = 15
long_loss = -22
short_benifit =15
short_loss = -22
max_drop = 5
min_drop = 5

long_diff = []
short_diff = []
total_diff = []
direction = []
changed = []
cumsum = []
pos_price = 0
min_price = 0 
max_price = 0
drop = 0

for row in tqdm(test_data.iterrows()):
    sign = row[1]['sign']

    roll_sum = sum(changed[-3:])
    roll_sum_1 = sum(changed[-4:-1])
    roll_diff = roll_sum - roll_sum_1
    roll_total = roll_sum+roll_diff

    # if roll_total >= 45 and roll_total <= 150:
    #     if sign == -1:
    #         sign *= -1
    # elif roll_total <= -45 and roll_total >= -150:
    #     if sign == 1:
    #         sign *= -1
    
    if pos == 0:
        if sign == 1:
            pos = 1
            pos_price = row[1]['close']+1
            max_price = pos_price
            
        elif sign == -1:
            pos = -1
            pos_price = row[1]['close']-1
            min_price = pos_price
            
    
    elif pos == 1:
        # 用high和low计算收盘
        max_close = row[1]['high'] -1
        min_close = row[1]['low'] + 1
        diff = max_close - pos_price
        max_price = max(max_price, row[1]['high'])
        drop = max_price - min_close
        if diff>=long_benifit and drop >= max_drop:
            long_diff.append(diff)
            pos = 0
            total_diff.append(diff)
            direction.append(1)
            changed.append(diff)
            cumsum.append(sum(changed))
        elif diff<= long_loss:
            long_diff.append(diff)
            pos = 0  
            total_diff.append(diff)
            direction.append(1)
            changed.append(diff)
            cumsum.append(sum(changed))
    elif pos == -1:
        # 用high和low计算收盘
        max_close = row[1]['high'] -1
        min_close = row[1]['low'] + 1
        diff = pos_price - min_close
        min_price = min(min_price, row[1]['low'])
        drop = max_close - min_price
        if diff>=short_benifit and drop >= min_drop:
            short_diff.append(diff)
            pos = 0
            total_diff.append(diff)
            direction.append(-1)
            changed.append(-1*diff)
            cumsum.append(sum(changed))
        elif diff<= short_loss:
            short_diff.append(diff)
            pos = 0
            total_diff.append(diff)
            direction.append(-1)
            changed.append(-1*diff)
            cumsum.append(sum(changed))
print(sum(long_diff),len(long_diff))
print(sum(short_diff),len(short_diff))

for i,j in zip(direction,total_diff):
    print(i,j)

test_dict = {'direction':direction,'total_diff':total_diff}
test_pd = pd.DataFrame(test_dict)
test_pd['changed'] = test_pd['direction']*test_pd['total_diff']
# tc = test_pd['changed'].cumsum()
# test_pd['total_changed'] = tc
test_pd['roll_changed'] = test_pd['changed'].rolling(3).sum()
test_pd['diff_changed'] = test_pd['roll_changed'].diff()
test_pd[:50]
