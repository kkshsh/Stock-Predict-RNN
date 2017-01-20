import numpy as np
import pymongo

from v4.db.db_connect import DBConnectManage
from v4.config import config

logger = config.get_logger(__name__)


class ReadDB:
    """field中的数据，日期默认会进行获取，不用额外放入fields中"""

    def __init__(self, fields=["openPrice", "closePrice", "highestPrice", "lowestPrice"]):
        self.client = DBConnectManage()
        self.collection = self.client.get_collection()
        self.read_count = 0
        self.fields = fields
        # warn: 这里一定要copy一份，而且需要用临时变量先存下来，否则append之后是None， 因为append成功之后返回的是none

    def read_one_stock_data(self, code, end_date=None, limit=None):
        logger.info("read count == %d", self.read_count)
        self.read_count += 1
        if limit is not None and (end_date is not None):
            total_count = self.collection.find({"ticker": code, "isOpen": 1, "tradeDate": {"$lte": end_date}}).count()
            dbData = self.collection.find({"ticker": code, "isOpen": 1, "tradeDate": {"$lte": end_date}}).sort("tradeDate", pymongo.ASCENDING) \
                .limit(limit).skip(total_count - limit)
        else:
            dbData = self.collection.find({"ticker": code, "isOpen": 1}).sort("tradeDate", pymongo.ASCENDING)

        data = []
        date_range = []
        for dataDict in dbData:
            data_tmp = []
            """过滤掉异常数据,日期默认就会获取，且需放在最后一个"""
            flag = True
            for field in self.fields:
                value = float(dataDict[field])
                if value < 0.001:
                    flag = False
                    break
                data_tmp.append(value)
            if flag:
                data.append(data_tmp)
                date_range.append(dataDict["tradeDate"])

        count = len(data)
        logger.info("stock code == %s, count == %d", code, count)
        return np.array(data), date_range

    def destory(self):
        self.client.close()


if __name__ == '__main__':
    print("train data>>>>>>")
