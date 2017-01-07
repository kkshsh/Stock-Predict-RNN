"""
use softmax regression and read data from mongodb
"""
import logging

import numpy as np
import tensorflow as tf
from v1.mongodb.util.datahandle import DataHandle
from v1.mongodb.util.readmongodb import ReadDB
from v1.quant.config.config import Option

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='/home/daiab/log/quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)


def batch(batch_size, data=None, target=None, rate=None, softmax=None, shuffle=False):
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
        yield data[excerpt], target[excerpt], rate[excerpt], softmax[excerpt]



class LstmModel:
    def __init__(self, session):
        self._session = session
        self._option = Option()

        self.allStockCode = [1] #readallcode()
        self.dataHandle = DataHandle(self._option.timeStep)
        self.readDb = ReadDB(datahandle=self.dataHandle)

    def updateData(self, code):
        self.readDb.readOneStockData(code)
        self.trainData = self.dataHandle.trainData
        self.target = self.dataHandle.target
        self.rate = self.dataHandle.rate
        self.softmax = self.dataHandle.softmax
        self.days = self.target.shape[0]
        self.testDays = (int)(self.days / 200)
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
        rate = self.rate[day:day + 1, :]
        return np.reshape(rate, [1, 1])

    def getOneEpochSoftmax(self, day):
        softmax = self.softmax[day:day + 1, :]
        return softmax

    def buildGraph(self):
        option = self._option
        self.oneTrainData = tf.placeholder(tf.float32, [None, option.timeStep, 5])
        self.targetPrice = tf.placeholder(tf.float32, [None, 2])
        cell = tf.nn.rnn_cell.BasicLSTMCell(option.hiddenCellNum, forget_bias=option.forget_bias,
                                           input_size=[option.batchSize, option.timeStep, option.hiddenCellNum])
        # cell = tf.nn.rnn_cell.BasicLSTMCell(option.hiddenCellNum,
        #                                    input_size=[option.batchSize, option.timeStep, option.hiddenCellNum])
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=option.keepProp)

        cell_2 = tf.nn.rnn_cell.MultiRNNCell([cell] * option.hiddenLayerNum)
        val, self.states = tf.nn.dynamic_rnn(cell_2, self.oneTrainData, dtype=tf.float32)

        # self.val = tf.transpose(val, [1, 0, 2])
        # self.lastTime = tf.gather(self.val, self.val.get_shape()[0] - 1)
        dim = option.timeStep * option.hiddenCellNum
        self.val = tf.reshape(val, [-1, dim])

        self.weight = tf.Variable(tf.truncated_normal([dim, option.outputCellNum], dtype=tf.float32), name='weight')
        self.bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1, option.outputCellNum]), name='bias')

        self.medianValue = tf.nn.relu(tf.matmul(self.val, self.weight) + self.bias)
        self.weight_2 = tf.Variable(tf.truncated_normal([option.outputCellNum, 2], dtype=tf.float32), name='weight_2')
        self.bias_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1, 2]), name='bias_2')

        self.predictPrice = tf.matmul(self.medianValue, self.weight_2) + self.bias_2

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predictPrice, self.targetPrice))
        self.minimize = tf.train.AdamOptimizer(learning_rate=option.learningRate).minimize(self.cross_entropy)
        self._session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def trainModel(self):
        option = self._option
        for i in range(len(self.allStockCode)):
            code = self.allStockCode.pop(0)
            self.updateData(code)
            """每次进入新的code时，重新init所有参数，开始全新一轮"""
            # self._session.run(tf.initialize_all_variables())
            # self.saver = tf.train.Saver()

            for epoch in range(option.epochs):
                logger.info("stockCode == %d, epoch == %d", code, epoch)
                batchData = batch(option.batchSize,
                                  self.trainData[:self.trainDays],
                                  self.target[:self.trainDays],
                                  self.rate[:self.trainDays],
                                  self.softmax[:self.trainDays], shuffle=True)
                feedDict = {}
                for oneEpochTrainData, _, _, softmax in batchData:
                    feedDict = {self.oneTrainData: oneEpochTrainData, self.targetPrice: softmax}
                    self._session.run(self.minimize, feed_dict=feedDict)
                    # print(self._session.run(self.val, feed_dict=feedDict).shape)

                if len(feedDict) != 0:
                    crossEntropy = self._session.run(self.cross_entropy, feed_dict=feedDict)
                    logger.info("crossEntropy == %f", crossEntropy)
                self.test()

            if option.is_save_file:
                self.saver.save(self._session, "/home/daiab/ckpt/code-%s.ckpt" % code)
                logger.info("save file code-%s", code)
        if option.loop_time > 1:
            option.loop_time -= 1
            self.allStockCode = readallcode()
            self.trainModel()

    def test(self):
        count, right = 0, 0
        logger.info("test begin ......")
        for day in range(self.trainDays, self.days - 1):
            trainData = [self.getOneEpochTrainData(day)]
            predictPrice = self._session.run(self.predictPrice,
                                             {self.oneTrainData: trainData})

            realPrice = self.getOneEpochSoftmax(day)

            if np.argmax(predictPrice) == np.argmax(realPrice):
                right += 1
            #     print("predict price right %s" % predictPrice)
            #     print("softmax %s", realPrice)
            # else:
            #     print("predict price error %s" % predictPrice)
            #     print("softmax %s", realPrice)

            count += 1

        count = 1 if count == 0 else count
        rightRatio = right / count
        logger.info("test right ratio >>>>>>>>>>>>>>>>>>>>>>>> %f", rightRatio)



def run():
    with tf.Graph().as_default(), tf.Session() as session:
        lstmModel = LstmModel(session)
        lstmModel.buildGraph()
        lstmModel.trainModel()
        session.close()
        lstmModel.readDb.destory()

if __name__ == '__main__':
    run()



