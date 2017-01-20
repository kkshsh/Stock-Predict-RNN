from pymongo import MongoClient
import json
import pandas as pd

client = MongoClient('db://localhost:27017/')
db = client.quant
collection = db.day_price
stock_code = '000001'

def insertByCsv():
    path = "/home/daiab/code/ml/something-interest/csv_data/000001.csv"
    data = pd.read_csv(path, index_col=0)
    rows = data.shape[0]
    print("rows >>>> %d" % rows)
    data['stock_code'] = pd.Series([stock_code] * rows, index=data.index)
    data['date'] = pd.Series(data.index.values, index=data.index)
    print(data[1:4])
    # print(json.loads(data.to_json(orient='records')))
    collection.insert(json.loads(data.to_json(orient='records')))

insertByCsv()
client.close()