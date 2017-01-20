import pandas as pd
import json
from v3.db.db_connect import DBConnectManage

connect = DBConnectManage()
collection = connect.get_collection()
file_path = "/home/daiab/Public/update_data/download-2016-12-03.csv"
data = pd.read_csv(file_path, index_col=0)
json_data = json.loads(data.to_json(orient='records'))
print(data.iloc[:3])

i = 0
for one in json_data:
    trade_date = one["tradeDate"]
    ticker = one["ticker"]
    count = collection.find({"tradeDate": trade_date, "ticker": ticker}).count()
    if count == 0:
        collection.insert(one)
    i += 1

print("insert into %d items" % i)

connect.close()
