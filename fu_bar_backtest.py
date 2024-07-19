import numpy as np
import pandas as pd
import time
from multiprocessing import Pool,cpu_count,Value,Manager
import multiprocessing
import os
import talib as ta
from tqdm import tqdm
import json
from datetime import datetime
from multiprocessing import Pool,cpu_count
import multiprocessing

def get_data(path,start=20000,period=1):
    origion = pd.read_csv(path)
    origion['datetime'] = pd.to_datetime(origion['datetime'])
    origion.set_index('datetime',inplace=True)
    origion = origion.dropna(axis=0)
    data = origion.resample(f'{period}min').agg(
    {
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last'
    }).dropna()
    if start >0 :
        start = -1*start

    return data[start:]


def backtest(args,is_fitting = True):

    data = args[0]
    result = args[1]
    benifit = args[2]
    loss = args[3]
    drop = args[4]
    assert(benifit>0)
    assert(loss<0)
    assert(drop>0)

    data['sma'] = ta.SMA(data['close'],3)
    data['slope'] = ta.LINEARREG_SLOPE(data['sma'],3)
    h1 = data[(data['slope'] <= 0.5)].index
    l1 =  data[(data['slope'] >= -0.5)].index
    data['sign']=0
    data.loc[h1,'sign'] = 1
    data.loc[l1,'sign'] = -1
    test_data = data

    

    pos = 0
    long_diff = []
    short_diff = []
    total_diff = []
    direction = []
    changed = []
    pos_price = 0
    min_price = 0 
    max_price = 0

    for row in tqdm(test_data.iterrows()):
        sign = row[1]['sign']    
        if pos == 0:
            if sign == 1:
                pos = 1
                pos_price = row[1]['close']+1
                max_price = pos_price
                min_price = pos_price
                
            elif sign == -1:
                pos = -1
                pos_price = row[1]['close']-1
                min_price = pos_price
                max_price = pos_price
        
        elif pos == 1:
            # 用high和low计算收盘
            max_price = max(max_price, row[1]['high'])
            min_price = min(min_price,row[1]['low'])
            price_benifit = max_price - drop
            diff_0 = price_benifit - pos_price
            diff_1 = min_price - pos_price
            
            if price_benifit <= row[1]['high'] and price_benifit >= row[1]["low"] and  diff_0 >= benifit:
                long_diff.append(diff_0)
                pos = 0
                total_diff.append(diff_0)
                direction.append(1)
                changed.append(diff_0)
                # cumsum.append(sum(changed))
            elif diff_1 <= loss:
                long_diff.append(diff_1)
                pos = 0  
                total_diff.append(diff_1)
                direction.append(1)
                changed.append(diff_1)
                # cumsum.append(sum(changed))
                
        elif pos == -1:
            # 用high和low计算收盘
            max_price = max(max_price, row[1]['high'])
            min_price = min(min_price,row[1]['low'])
            price_benifit = min_price + drop
            diff_0 =  pos_price - price_benifit 
            diff_1 = pos_price - max_price
            
            if price_benifit <= row[1]['high'] and price_benifit >= row[1]["low"] and diff_0 >= benifit:
                short_diff.append(diff_0)
                pos = 0
                total_diff.append(diff_0)
                direction.append(-1)
                changed.append(-1*diff_0)

            elif diff_1<= loss:
                short_diff.append(diff_1)
                pos = 0
                total_diff.append(diff_1)
                direction.append(-1)
                changed.append(-1*diff_1)

    res = {"long_diff":sum(long_diff),"long_trades":len(long_diff),
            "short_diff":sum(short_diff),"short_trades":len(short_diff)}
    result.append({f"{benifit}_{loss}_{drop}":res})

def write_result(result):
    base_name = os.path.basename(__file__).split('.')[0]
    now = datetime.now().strftime('%Y-%m-%d %H-%M')
    file_name = 'D:/Code/python_project/bar_analysis/result/'+base_name +f'_{now}.jsonl'
    with open(file_name,'w') as f:
        for data in result:
            # 将每个数据对象转换为JSON字符串并写入文件
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    manager = Manager()
    result = manager.list()
    lock = manager.Lock()
    input_path = "D:/Code/jupyter_project/data_analysis/bar_analysis/bar_backtest/fu数据分析/fu88.csv"
    data = get_data(input_path)
    param =[(data,result,20,-22,11),(data,result,14,-18,8)]
    worker_num = 1
    pool = Pool(worker_num)
    pool.map(backtest, param)
    pool.close()
    pool.join()
    write_result(result)
