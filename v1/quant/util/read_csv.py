import numpy as np
import pandas as pd

class DataHandle:
    def __init__(self, filePath, timeStep):
        self.TIME_STEP = timeStep
        originData = self.readCsv(filePath)
        self.formatDataDim(originData)
        # print(self.trainData[:5])
        # print("==================")
        # print(self.datasource[0:60])

    def readCsv(self, file_path):
        csv = pd.read_csv(file_path, index_col=0)
        return csv.reindex(index=csv.index[::-1])

    def formatDataDim(self, data):
        data = self.concatenateData(data)
        self.data = self.normalization(data)
        self.days = self.data.shape[0]
        self.target = self.data[:, 1:2]
        # print("target >>>>>>>>>>>>>>>>>>>>")
        # print(self.target)
        self.trainData = self.buildSample(self.data)

    def concatenateData(self, data):
        data = np.concatenate(
            [data.open.values[:, np.newaxis],
             data.close.values[:, np.newaxis],
             data.high.values[:, np.newaxis],
             data.low.values[:, np.newaxis]], 1)
        # print("concat datasource==========>")
        # print(datasource)
        return data

    def normalization(self, data):
        # rows = datasource.shape[0]
        # norm = (csv_data - csv_data.min(axis=0)) / (csv_data.max(axis=0) - csv_data.min(axis=0))
        norm = (data - data.mean(axis=0)) / data.var(axis=0)
        return norm

    def buildSample(self, data):
        result = []
        rows = data.shape[0]
        for i in range(self.TIME_STEP, rows + 1):
            tmp = []
            tmp.append(data[i - self.TIME_STEP: i, :])
            result.append(np.array(tmp))
        return result

if __name__ == '__main__':
    dataHandle = DataHandle("/home/daiab/code/ml/something-interest/csv_data/000001-minute.csv", 20)
    # print(dataHandle.originData)
    print(dataHandle.data[-200:])
    print(dataHandle.target[-200:])