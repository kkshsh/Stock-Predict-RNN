import pandas as pd
import json
from v4.db.db_connect import DBConnectManage
from v4.config import config

connect = DBConnectManage()
collection = connect.get_collection()
data = pd.read_csv(config.update_uqer_csv_file_path, index_col=0)
json_data = json.loads(data.to_json(orient='records'))
print(data.iloc[:3])

i = 0
print("start to update ......")
for one in json_data:
    trade_date = one["tradeDate"]
    ticker = one["ticker"]
    count = collection.find({"tradeDate": trade_date, "ticker": ticker}).count()
    if count == 0:
        collection.insert(one)
        i += 1

print("over...insert into %d items" % i)

connect.close()
