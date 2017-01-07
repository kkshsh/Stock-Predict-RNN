import tushare as ts
import numpy as np
import matplotlib.pyplot as plt

price = ts.get_hist_data('000001', '2016-09-01')

index_price = price.low.values
index_price = index_price[::-1]
num = index_price.shape[0]
minus = np.concatenate([[index_price[0]], index_price[0: num - 1]], 0)
normalize = index_price / minus - 1
days = range(num)
plt.plot(days, normalize, 'r-')
#plt.show()
#plt.plot(days, index_price)
plt.show()
