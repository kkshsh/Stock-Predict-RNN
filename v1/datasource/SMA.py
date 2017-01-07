import numpy as np
import pandas as pd
import talib as ta
day = 20

p = pd.read_csv('2016.csv',index_col=0)
price = p.icol([0,1,2,3]).values
# print(price)
open_p = ta.SMA(price[:,0], day)[:,np.newaxis]
high_p = ta.SMA(price[:,1], day)[:,np.newaxis]
low_p = ta.SMA(price[:,2], day)[:,np.newaxis]
close_p = ta.SMA(price[:,3], day)[:,np.newaxis]

SMA = np.concatenate([open_p, high_p, low_p, close_p], 1)
SMA = SMA[day:]

result = pd.DataFrame(SMA)
result.to_csv('SMA.csv')

# print(result)

