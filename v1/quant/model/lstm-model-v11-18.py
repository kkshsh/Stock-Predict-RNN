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
        self.timeStep = timeStep
        originData = self.readCsv(filePath)
        self.formatDataDim(originData)
        # print("originData--------------")
        # print(originData.head(100))

    def readCsv(self, file_path):
        csv = pd.read_csv(file_path, index_col=0)
        return csv.reindex(index=csv.index[::-1])

    def formatDataDim(self, data):
        data = self.concatenateData(data)
        # zscore
        zscoreData = self.zscore(data)
        # rate
        rateNormData = self.rateNorm(data)
        self.target = zscoreData[self.timeStep:, 1:2]
        # print("target %s" % self.target[:5])
        self.ratio = rateNormData[self.timeStep:, 1:2]
        # print("ratio %s" % self.ratio[:5])
        self.softmax = self.softmaxTarget(self.ratio)
        # print("softmax %s" % self.softmax[:5])
        self.days = self.target.shape[0]
        self.trainData = self.buildSample(zscoreData)[:-1]


    def concatenateData(self, data):
        data = np.concatenate(
            [data.open.values[:, np.newaxis],
             data.close.values[:, np.newaxis],
             data.high.values[:, np.newaxis],
             data.low.values[:, np.newaxis]], 1)
        # print("concat datasource==========>")
        # print(datasource)
        return data

    def zscore(self, data):
        # rows = datasource.shape[0]
        # norm = (csv_data - csv_data.min(axis=0)) / (csv_data.max(axis=0) - csv_data.min(axis=0))
        norm = (data - data.mean(axis=0)) / data.var(axis=0)
        return norm

    def rateNorm(self, data):
        norm = np.zeros_like(data)
        for i in range(data.shape[0] - 1, 0, -1):
            norm[i] = (data[i] / data[i - 1]) - 1
        norm[0] = 0
        return norm

    def softmaxTarget(self, ratio):
        softmax = np.zeros(shape=[ratio.shape[0], 2])
        #[1, 0] refer to increment, [0, 1] refer to decrement
        softmax[np.where(ratio[:, 0] >= 0), 0] = 1
        # print(softmax[:5])
        softmax[np.where(ratio[:, 0] < 0), 1] = 1
        # print(softmax[:5])
        return softmax

    def buildSample(self, data):
        result = []
        rows = data.shape[0]
        for i in range(self.timeStep, rows + 1):
            result.append(np.array(data[i - self.timeStep: i, :]))
        return np.array(result)

def batch(batch_size, data=None, target=None, ratio=None, softmax=None, shuffle=False):
    assert len(data) == len(target)
    if shuffle:
        indices = np.arange(len(data), dtype=np.int32)
        # indices = range(len(datasource))
        np.random.shuffle(indices)
    for start_idx in range(0, len(data) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield data[excerpt], target[excerpt], ratio[excerpt], softmax[excerpt]



class LstmModel:
    def __init__(self):
        filePath = '/home/daiab/code/ml/something-interest/csv_data/601901.csv'
        # filePath = '/home/daiab/code/ml/something-interest/datasource/601988.csv'
        # filePath = '/home/daiab/code/ml/something-interest/datasource/000068.csv'
        self.timeStep = 19
        self.hiddenNum = 50
        self.epochs = 200
        self._session = tf.Session()

        dataHandle = DataHandle(filePath, self.timeStep)
        self.trainData = dataHandle.trainData
        self.target = dataHandle.target
        self.ratio = dataHandle.ratio
        self.softmax = dataHandle.softmax

        self.days = self.target.shape[0]
        self.testDays = (int)(self.days / 6)
        print("all days is %d" % self.days)
        self.trainDays = self.days - self.testDays
        self.isPlot = True
        self.batchSize = 10
        del dataHandle

    # 当前天前timeStep天的数据，包含当前天
    def getOneEpochTrainData(self, day):
        assert day >= 0
        # print("get_one_epoch_data >>>>>>>>>>>>>>")
        # print(self.trainData[index])
        return self.trainData[day]

    # 当前天后一天的数据
    def getOneEpochTarget(self, day):
        target = self.target[day:day + 1, :]
        return np.reshape(target, [1, 1])

    def getOneEpochRatio(self, day):
        ratio = self.ratio[day:day + 1, :]
        return np.reshape(ratio, [1, 1])

    def buildGraph(self):
        self.oneTrainData = tf.placeholder(tf.float32, [None, self.timeStep, 4])
        self.targetPrice = tf.placeholder(tf.float32, [None, 1])
        cell = tf.nn.rnn_cell.LSTMCell(self.hiddenNum)

        cell_2 = tf.nn.rnn_cell.MultiRNNCell([cell] * 1)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
        val, self.states = tf.nn.dynamic_rnn(cell_2, self.oneTrainData, dtype=tf.float32)
        # tf.nn.xw_plus_b()
        # tf.get_variable()

        self.val = tf.transpose(val, [1, 0, 2])
        self.lastTime = tf.gather(self.val, self.val.get_shape()[0] - 1)

        self.weight = tf.Variable(tf.truncated_normal([self.hiddenNum, 1], dtype=tf.float32))
        self.bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.batchSize, 1]))
        self.predictPrice = tf.matmul(self.lastTime, self.weight) + self.bias
        self.diff = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.predictPrice, self.targetPrice)))

        inse = tf.reduce_sum(self.predictPrice * self.targetPrice)
        l = tf.reduce_sum(self.predictPrice * self.predictPrice)
        r = tf.reduce_sum(self.targetPrice * self.targetPrice)
        self.dice = 2 * inse / (l + r)

        self.minimize = tf.train.AdamOptimizer().minimize(self.diff - self.dice)

    with tf.device("/cpu:0"):
        def trainModel(self):
            self._session.run(tf.initialize_all_variables())
            for epoch in range(self.epochs):
                print("epoch %i >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" % epoch)
                batchData = batch(self.batchSize,
                                  self.trainData[:self.trainDays],
                                  self.target[:self.trainDays],
                                  self.ratio[:self.trainDays],
                                  self.softmax[:self.trainDays], shuffle=False)
                diffSum = 0
                count = 1
                for oneEpochTrainData, _, ratio, _ in batchData:
                    # print(len(oneEpochTrainData))
                    # print("day %d" % day)

                    dict = {self.oneTrainData: oneEpochTrainData, self.targetPrice: ratio}

                    self._session.run(self.minimize, feed_dict=dict)

                    diff = self._session.run(self.diff, feed_dict=dict)

                    diffSum += diff
                    count += 1

                    if count % 20 == 0:
                        predictPrice = self._session.run(self.predictPrice, feed_dict=dict)
                        dice = self._session.run(self.dice, feed_dict=dict)
                        print("...................................................")
                        # print("predictPrice is %s" % predictPrice)
                        # print("real price is %s" % ratio)
                        print("diff is %s" % diff)
                        print("dice is %s" % dice)
                        # print("state is %s" % states)
                        # print(states.shape)
                print("diff mean >>>>>>>>>>>>>> %f" % (diffSum / count))
                if epoch % 20 == 0:
                    self.test()

    with tf.device("/cpu:1"):
        def test(self):
            predict = []
            real = []
            dayIndex = []

            for day in range(self.trainDays, self.days - 1):
                trainData = [self.getOneEpochTrainData(day)]
                # target = self.getOneEpochTarget(day)

                predictPrice = self._session.run(self.predictPrice,
                                                 {self.oneTrainData: trainData})[0, 0]

                realPrice = self.getOneEpochRatio(day)[0, 0]

                predict.append(predictPrice)
                real.append(realPrice)
                dayIndex.append(day)
            if self.isPlot:
                self.plotLine(dayIndex, predict, real)

    with tf.device("/cpu:3"):
        def plotLine(self, days, predict, real):
            plt.ylabel("close")
            plt.grid(True)
            plt.plot(days, predict, 'r-')
            plt.plot(days, real, 'b-')
            plt.show()


    def run(self):
        self.buildGraph()
        self.trainModel()
        self.test()
        self._session.close()

if __name__ == '__main__':
    lstmModel = LstmModel()
    lstmModel.run()



