"""
使用多层lstm神经层,重新使用减去均值除以标准差的方式正则化数据,仅预测close price
尝试测试多个stock
"""
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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



class LstmModel:
    def __init__(self):
        filePath = '/home/daiab/code/ml/something-interest/csv_data/000001-minute.csv'
        # filePath = '/home/daiab/code/ml/something-interest/datasource/601988.csv'
        # filePath = '/home/daiab/code/ml/something-interest/datasource/000068.csv'
        self.TIME_STEP = 20
        self.NUM_HIDDEN = 50
        self.epochs = 200
        self.testDays = 40
        self._session = tf.Session()
        dataHandle = DataHandle(filePath, self.TIME_STEP)
        self.data = dataHandle.data
        self.trainData = dataHandle.trainData
        self.days = dataHandle.days
        self.trainDays = self.days - self.testDays
        self.isPlot = True

    # 当前天前timeStep天的数据，包含当前天
    def getOneEpochTrainData(self, day):
        assert day >= self.TIME_STEP - 1
        index = day - (self.TIME_STEP - 1)
        # print("get_one_epoch_data >>>>>>>>>>>>>>")
        # print(self.trainData[index])
        return self.trainData[index]

    # 当前天后一天的数据
    def getOneEpochTarget(self, day):
        target = self.data[day + 1][1:2]
        # target = np.hsplit(target, [1])[1]
        # print("get_one_epoch_target >>>>>>>>>>>>>>")
        # print(np.array(target))
        return target[np.newaxis, :]

    def buildGraph(self):
        self.oneTrainData = tf.placeholder(tf.float32, [1, self.TIME_STEP, 4])
        # self.train_target = tf.placeholder(tf.float32, [1, 4])
        self.targetPrice = tf.placeholder(tf.float32, [1, 1])
        cell = tf.nn.rnn_cell.LSTMCell(self.NUM_HIDDEN)
        # self.val = tf.transpose(val, [1, 0, 2])
        # self.last_time = tf.gather(self.val, self.TIME_STEP - 1)

        cell_2 = tf.nn.rnn_cell.MultiRNNCell([cell] * 1)
        val, _ = tf.nn.dynamic_rnn(cell_2, self.oneTrainData, dtype=tf.float32)

        self.val = tf.transpose(val, [1, 0, 2])
        self.lastTime = tf.gather(self.val, self.TIME_STEP - 1)


        self.weight = tf.Variable(tf.truncated_normal([self.NUM_HIDDEN, 1], dtype=tf.float32))
        self.bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1, 1]))
        self.predictPrice = tf.matmul(self.lastTime, self.weight) + self.bias

        self.diff = tf.sqrt(tf.reduce_mean(tf.square(self.predictPrice - self.targetPrice)))

        self.minimize = tf.train.AdamOptimizer().minimize(self.diff)

    def trainModel(self):
        init_op = tf.initialize_all_variables()
        self._session.run(init_op)
        self.rightNumArr = []
        for epoch in range(self.epochs):
            print("epoch %i >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" % epoch)
            rightNum = 0
            for day in range(self.TIME_STEP, self.trainDays):
                realPrice = self.getOneEpochTarget(day)
                oneEpochTrainData = self.getOneEpochTrainData(day)

                self._session.run(self.minimize,
                                  {self.oneTrainData: oneEpochTrainData,
                                   self.targetPrice: realPrice})

                predictPrice = self._session.run(self.predictPrice,
                                                 {self.oneTrainData: oneEpochTrainData,
                                                  self.targetPrice: realPrice})

                diff = self._session.run(self.diff,
                                         {self.oneTrainData: oneEpochTrainData,
                                          self.targetPrice: realPrice})

                yesterdayRealPrice = self.getOneEpochTarget(day - 1)

                trend = (predictPrice - yesterdayRealPrice) * (realPrice - yesterdayRealPrice)

                if trend[0][0] > 0:
                    rightNum += 1

                if day % 100 == 0:
                    print("...................................................")
                    print("predictPrice is %s" % predictPrice)
                    print("real price is %s" % self.getOneEpochTarget(day))
                    print("diff is %s" % diff)

            rightRatio = rightNum / (self.trainDays - self.TIME_STEP)
            print("right ratio is %f" % rightRatio)

            if epoch % 40 == 0:
                self.rightNumArr.append(self.test())
        print("rightNumArr is %s" % self.rightNumArr)

    def test(self):
        predict = np.array([[0]])
        real = np.array([[0]])
        dayIndex = [self.trainDays - 1]
        rightNum = [0]
        for day in range(self.trainDays, self.days - 1):
            predictPrice = self._session.run(self.predictPrice,
                                              {self.oneTrainData: self.getOneEpochTrainData(day),
                                               self.targetPrice: self.getOneEpochTarget(day)})
            realPrice = self.getOneEpochTarget(day)
            yesterdayRealPrice = self.getOneEpochTarget(day - 1)


            # check whether trend between predict and real is consistent
            trend = (predictPrice - yesterdayRealPrice)[0] * (realPrice - yesterdayRealPrice)[0]
            trend[trend > 0] = 1
            trend[trend <= 0] = 0
            rightNum += trend

            # print(predict_price)
            predict = np.concatenate([predict, predictPrice], 0)
            # print(predict)
            real = np.concatenate([real, realPrice], 0)
            # print(real)
            dayIndex.append(day)
        # print(predict[:, 0])
        if self.isPlot:
            self.plotLine(dayIndex[1:], predict[1:, :], real[1:, :])
        print("rightNum is %s" % rightNum)
        return rightNum

    def plotLine(self, days, predict, real):
        plt.ylabel("close")
        plt.grid(True)
        plt.plot(days, predict[:, 0], 'r-')
        plt.plot(days, real[:, 0], 'b-')
        plt.show()


    def run(self):
        self.buildGraph()
        self.trainModel()
        self.test()
        self._session.close()

if __name__ == '__main__':
    lstmModel = LstmModel()
    lstmModel.run()



