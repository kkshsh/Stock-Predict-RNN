import tushare as ts
# https://github.com/waditu/tushare

#hist = ts.get_hist_data('600848')
hist = ts.get_hist_data('000001')
print(hist.ix[1:3])
print(hist.close)
