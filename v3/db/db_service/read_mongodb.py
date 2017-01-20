from pymongo import MongoClient
import pymongo
from v3.service.data_preprocess import DataPreprocess
import logging
import pandas as pd
from db.db_connect import DBConnectManage

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='/home/daiab/log/quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)


class ReadDB:
    """field中的数据，日期默认会进行获取，不用额外放入fields中"""
    def __init__(self, data_preprocess, fields=["openPrice", "closePrice", "highestPrice", "lowestPrice"]):
        self.client = DBConnectManage()
        self.collection = self.client.get_collection()
        self.data_preprocess = data_preprocess
        self.read_count = 0
        self.fields = fields
        # warn: 这里一定要copy一份，而且需要用临时变量先存下来，否则append之后是None， 因为append成功之后返回的是none
        self.columns = fields.copy()
        self.columns.append("tradeDate")

    def read_one_stock_data(self, code):
        logger.info("read count == %d", self.read_count)
        self.read_count += 1
        dbData = self.collection.find({"ticker": code, "isOpen": 1}).sort("tradeDate", pymongo.ASCENDING)
        data = []
        for dataDict in dbData:
            tmp = []
            """过滤掉异常数据,日期默认就会获取，且需放在最后一个"""
            flag = True
            for field in self.fields:
                value = float(dataDict[field])
                if value < 0.001:
                    flag = False
                    break
                tmp.append(value)
            if flag:
                tmp.append(dataDict["tradeDate"])
                data.append(tmp)

        count = len(data)
        logger.info("stock code == %s, count == %d", code, count)
        data = pd.DataFrame(data, columns=self.columns).set_index("tradeDate", append=False)
        # print("origin mongodb data>>>>>>")
        # print(data.loc['2016-11-10':'2016-11-18'])
        self.data_preprocess.process(data)

    def destory(self):
        self.client.close()

if __name__=='__main__':
    data_process = DataPreprocess(2)
    readData = ReadDB(data_process)
    readData.read_one_stock_data(1)
    print("train data>>>>>>")
    for date in data_process.train_data.loc['2016-11-11':'2016-11-17']:
        print("date time == %s" % date)
        print(data_process.train_data[date])
    print("target data>>>>>>")
    print(data_process.target.loc['2016-11-11':'2016-11-17'])
    print("rate data>>>>>>")
    print(data_process.rate.loc['2016-11-11':'2016-11-17'])
    print("softmax data>>>>>>")
    print(data_process.softmax.loc['2016-11-11':'2016-11-17'])

