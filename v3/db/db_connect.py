from pymongo import MongoClient


class DBConnectManage:
    def __init__(self):
        self._client = MongoClient('mongodb://localhost:27017/')
        db = self._client.quant
        self._collection = db.uqer

    def get_collection(self):
        return self._collection

    def close(self):
        self._client.close()