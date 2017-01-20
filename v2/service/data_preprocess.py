import logging

import numpy as np
from v2.config.config import Option

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='/home/daiab/log/quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)


class DataPreprocess:
    def __init__(self, timeStep):
        self.timeStep = timeStep
        self.ZScore = ZScore()
        self.SoftmaxHandle = SoftmaxHandle()
        self.RateNorm = RateNorm()
        self.BuildSerial = BuildSerial(timeStep)
        self._option = Option()

    """
    input: numpy type array, shape like [sample_size, variable_size]
    """
    def process(self, origin_data):
        if self._option.train_data_type == "zscore":
            self.trainData = self.BuildSerial.generate_zscore_serial(origin_data)[:-1, :, :]
        elif self._option.train_data_type == "rate":
            self.trainData = self.BuildSerial.generate_rate_serial(origin_data)[:-1, :, :]
            # print(self.trainData[:3])
        else:
            raise Exception("train data type error")

        if self._option.predict_type == "open":
            origin_target_data = origin_data[:, 0:1]
        elif self._option.predict_type == "close":
            origin_target_data = origin_data[:, 1:2]
        elif self._option.predict_type == "high":
            origin_target_data = origin_data[:, 2:3]
        elif self._option.predict_type == "low":
            origin_target_data = origin_data[:, 3:4]
        else:
            raise Exception("predict type error")

        self.target = self.ZScore.convert_to_zscore(origin_target_data)[self.timeStep:]
        self.rate = self.RateNorm.convert_to_rate(origin_target_data)[self.timeStep:]
        self.softmax = self.SoftmaxHandle.general_softmax_target(origin_target_data)[self.timeStep:]
        self.days = self.target.shape[0]

    def validate(self):
        origin_data = np.array([[1, 2],
                                [3, 4],
                                [1, 2],
                                [3, 4],
                                [1, 2],
                                [3, 4]])
        z_score_data = np.array([[-1, -1],
                                 [1, 1],
                                 [-1, -1],
                                 [1, 1],
                                 [-1, -1],
                                 [1, 1]])[self.timeStep:, 0:1]
        rate = np.array([[0, 0],
                         [2, 1],
                         [-0.666666, -0.5],
                         [2, 1],
                         [-0.666666, -0.5],
                         [2, 1]])[self.timeStep:, 0:1]
        softmax = np.array([[0, 0],
                            [1, 0],
                            [0, 1],
                            [1, 0],
                            [0, 1],
                            [1, 0]])[self.timeStep:]
        train_data = np.array([[[-1, -1],
                                [1, 1]],
                              [[1, 1],
                               [-1, -1]],
                              [[-1, -1],
                               [1, 1]],
                              [[1, 1],
                               [-1, -1]],
                              [[-1, -1],
                               [1, 1]]])[:-1]
        self._option.predict_type = "open"
        self.process(origin_data)
        print("trainData: real:\n%s \n calculate:\n%s" % (train_data, datahandle.trainData))
        print("target: real:\n%s \n calculate:\n%s" % (z_score_data, datahandle.target))
        print("rate: real:\n%s \n calculate:\n%s" % (rate, datahandle.rate))
        print("softmax: real:\n%s \n calculate:\n%s" % (softmax, datahandle.softmax))


class ZScore:
    def __init__(self):
        pass

    """
    input: numpy array type data, shape like [sample_size, variable_size]
    output: [sample_size, variable_size]
    """
    def convert_to_zscore(self, data):
        return (data - data.mean(axis=0)) / data.std(axis=0)

    def validate(self):
        pass


class SoftmaxHandle:
    def __init__(self):
        pass

    """
    input: numpy array type data, shape like [sample_size, 1], just handle one variable per time
    output: [sample_size, 2], the first sample will handled to [1, 0] or [0, 1] by probability,
            because it have no reference for contrast
            increase will denote as [1, 0], instead decrease will denote as [0, 1]
    for example:
    input: [[2],
            [3],
            [5],
            [4]]
    output:[[0, 0],
            [1, 0],
            [1, 0],
            [0, 1]]
    """
    def general_softmax_target(self, array):
        sample_size = array.shape[0]
        softmax = np.zeros(shape=[sample_size, 2])
        for row in range(1, sample_size):
            if array[row][0] >= array[row - 1][0]:
                softmax[row][0] = 1
            else:
                softmax[row][1] = 1
        assert softmax.shape[0] == sample_size
        return softmax

    def validate(self):
        pass


class RateNorm:
    def __init__(self):
        pass

    """
    input: numpy array type data, shape like [sample_size, variable_size]
            the first sample will handled to be [1, 1, ..., 1]
    output: [sample_size, variable_size]
    """
    def convert_to_rate(self, data):
        sample_size, variable_size = data.shape[0], data.shape[1]
        rate = np.ones_like(data, dtype=float)
        for row in range(sample_size - 1, 0, -1):
            row_rate = data[row] / data[row - 1] - 1
            rate[row] = row_rate
        assert rate.shape[0] == sample_size
        return rate

    def validate(self):
        pass

class BuildSerial:
    """
    length: how many samples per serial
    """
    def __init__(self, length):
        self.length = length

    """
    input: numpy array type data, shape like [sample_size, variable_size]
            the input will be normalized to z-score for preprocess
    output: [sample_size - length + 1, sample_size, variable_size]
    example: length = 2
    input: [[1, 2],
            [2, 3],
            [3, 4],
            [4, 5]]
    output: [[[1, 2],
              [2, 3]],
             [[2, 3],
              [3, 4]],
             [[3, 4],
              [4, 5]]]
    """
    def generate_zscore_serial(self, data):
        data = ZScore().convert_to_zscore(data)
        sample_size = data.shape[0]
        result = []
        for row in range(sample_size - self.length + 1):
            result.append(data[row: row + self.length])
        assert len(result) == sample_size - self.length + 1
        return np.array(result)

    """like generate_zscore_serial"""
    def generate_rate_serial(self, data):
        data = RateNorm().convert_to_rate(data)
        sample_size = data.shape[0]
        result = []
        for row in range(sample_size - self.length + 1):
            result.append(data[row: row + self.length])
        assert len(result) == sample_size - self.length + 1
        return np.array(result)

    def validata(self):
        pass


if __name__ == "__main__":
    datahandle = DataPreprocess(2)
    datahandle.validate()


