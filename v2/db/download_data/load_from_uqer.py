import os
import pandas as pd
import json
from pymongo import MongoClient

client = MongoClient('db://localhost:27017/')
db = client.quant
collection = db.uqer

dir = '/home/daiab/Public/zip/data'
allcode = []
days = []
for dirpath, dirnames, filenames in os.walk(dir):
    for filename in filenames:
        data = pd.read_csv(os.path.join(dirpath, filename), index_col=0)
        # if data is null, then db will throw exception
        rows = data.shape[0]
        if rows > 0:
            allcode.append(int(filename.split(".")[0]))
            days.append(rows)

            json_data = json.loads(data.to_json(orient='records'))
            collection.insert(json_data)
            print("filename is %s, rows = %d " % (filename, data.shape[0]))

allcode = pd.Series(allcode, name="code", dtype=str).to_frame()
allcode['days'] = pd.Series(days, name="days", dtype=int)
allcode.to_csv("all_code.csv")
print("over")
client.close()
