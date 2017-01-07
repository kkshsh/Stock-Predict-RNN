from pymongo import MongoClient
import tushare as ts
import json
import pandas as pd

client = MongoClient('db://localhost:27017/')
db = client.quant
collection = db.day_price

# price = ts.get_hist_data('000001')
# collection.insert(json.loads(price.to_json(orient='records')))

count = collection.find({"stock_code": '000001'}).count()
print(count)

result = collection.find({"stock_code": '000001'}).sort("date").limit(2)
data = []
for i in result:
    tmp = []
    tmp.append(i["open"])
    tmp.append(i["close"])
    tmp.append(i["high"])
    tmp.append(i["low"])
    data.append(tmp)
print(data)


client.close()


