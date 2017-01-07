"""
use softmax regression and read data from mongodb
"""
import logging

import numpy as np
import tensorflow as tf
from db import DataHandle
from db import ReadDB

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='/home/daiab/log/quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)


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
    def __init__(self, session):
        self.timeStep = 15
        self.hiddenNum = 400
        self.epochs = 200
        self._session = session

        self.allStockCode = ['000001']
        self.dataHandle = DataHandle(self.timeStep)
        self.readDb = ReadDB(stockCodeList=self.allStockCode, datahandle=self.dataHandle)
        # 从数据库取出一次数据后，重复利用几次
        self.reuseTime = 1000
        self.batchSize = 50
        self.counter = {}
        # save current training stock
        self.currentStockCode = ""

    def updateData(self):
        self.readDb.read_one_stock_data()
        self.trainData = self.dataHandle.trainData
        self.target = self.dataHandle.target
        self.ratio = self.dataHandle.ratio
        self.softmax = self.dataHandle.softmax
        self.days = self.target.shape[0]
        self.testDays = (int)(self.days / 9)
        self.trainDays = self.days - self.testDays


    # 当前天前timeStep天的数据，包含当前天
    def getOneEpochTrainData(self, day):
        assert day >= 0
        return self.trainData[day]

    # 当前天后一天的数据
    def getOneEpochTarget(self, day):
        target = self.target[day:day + 1, :]
        return np.reshape(target, [1, 1])

    def getOneEpochRatio(self, day):
        ratio = self.ratio[day:day + 1, :]
        return np.reshape(ratio, [1, 1])

    def getOneEpochSoftmax(self, day):
        softmax = self.softmax[day:day + 1, :]
        return softmax

    def buildGraph(self):
        self.oneTrainData = tf.placeholder(tf.float32, [None, self.timeStep, 4])
        self.targetPrice = tf.placeholder(tf.float32, [None, 2])
        cell = tf.nn.rnn_cell.BasicRNNCell(self.hiddenNum)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=0.8)

        cell_2 = tf.nn.rnn_cell.MultiRNNCell([cell] * 5)
        val, self.states = tf.nn.dynamic_rnn(cell_2, self.oneTrainData, dtype=tf.float32)

        self.val = tf.transpose(val, [1, 0, 2])
        self.lastTime = tf.gather(self.val, self.val.get_shape()[0] - 1)

        self.weight = tf.Variable(tf.truncated_normal([self.hiddenNum, 100], dtype=tf.float32), name='weight')
        self.bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1, 100]), name='bias')

        self.medianValue = tf.matmul(self.lastTime, self.weight) + self.bias

        self.weight_2 = tf.Variable(tf.truncated_normal([100, 2], dtype=tf.float32), name='weight_2')
        self.bias_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1, 2]), name='bias_2')

        self.predictPrice = tf.matmul(self.medianValue, self.weight_2) + self.bias_2

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predictPrice, self.targetPrice))
        self.minimize = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(self.cross_entropy)
        self._session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def trainModel(self):
        for epoch in range(self.epochs):
            logger.info("epoch-epoch-epoch-epoch")
            time = 0
            for stockCode in self.allStockCode:
                logger.info("epoch == %d, time == %d", epoch, time)
                self.updateData()
                self.currentStockCode = stockCode
                for useTime in range(self.reuseTime):
                    logger.info("reuse time == %d", useTime)
                    batchData = batch(self.batchSize,
                                      self.trainData[:self.trainDays],
                                      self.target[:self.trainDays],
                                      self.ratio[:self.trainDays],
                                      self.softmax[:self.trainDays], shuffle=False)

                    dict = {}
                    for oneEpochTrainData, _, _, softmax in batchData:
                        dict = {self.oneTrainData: oneEpochTrainData, self.targetPrice: softmax}
                        # logger.info("dict == %s", dict)

                        self._session.run(self.minimize, feed_dict=dict)

                    if len(dict) != 0:
                        crossEntropy = self._session.run(self.cross_entropy, feed_dict=dict).sum()
                        logger.info("crossEntropy == %f", crossEntropy)

                    self.test()

            self.saver.save(self._session, "/home/daiab/ckpt/model-%s.ckpt" % epoch)
            logger.info("save file %s", epoch)

            self.removeBadStock()
            self.counter = {}

    def removeBadStock(self):
        logger.info("counter == \n%s" % self.counter)
        logger.info(len(self.allStockCode))
        stockRank = sorted(self.counter.items(), key = lambda x: x[1])
        removeNum = 20
        index = 0
        for stockCode, _ in stockRank:
            logger.info("remove stock code %s" % stockCode)
            if index >= removeNum:
                break
            self.allStockCode.remove(stockCode)
            index += 1
        logger.info("after remove the left stock code \n %s" % self.allStockCode)
        self.readDb.updateStockCode(self.allStockCode)


    def test(self):
        count, right = 0, 0
        logger.info("test begin ......")

        for day in range(self.trainDays, self.days - 1):
            trainData = [self.getOneEpochTrainData(day)]
            predictPrice = self._session.run(self.predictPrice,
                                             {self.oneTrainData: trainData})

            realPrice = self.getOneEpochSoftmax(day)

            if np.argmax(predictPrice) == np.argmax(realPrice): right += 1
            count += 1

        count = 1 if count == 0 else count
        rightRatio = right / count
        logger.info("test right ratio >>>>>>>>>>>>>>>>>>>>>>>> %f", rightRatio)
        self.counter[self.currentStockCode] = rightRatio



def run():
    with tf.Graph().as_default(), tf.Session() as session:
        lstmModel = LstmModel(session)
        lstmModel.buildGraph()
        lstmModel.trainModel()
        # lstmModel.test()
        session.close()
        lstmModel.readDb.destory()

if __name__ == '__main__':
    run()



