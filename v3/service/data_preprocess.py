import logging

import numpy as np
import pandas as pd
from v3.config.config import Option

"""
一定要注意，处理数据时避免在源数据上进行赋值,以及类型转化可能隐含的错误
如果时间跨度是5天，取'2015-01-10'的训练数据时返回['2015-01-06','2015-01-07','2015-01-08','2015-01-09','2015-01-10'],
也就是包含'2015-01-10'这一天的数据。但是取目标数据时返回['2015-01-11']，也就是第二天的数据。这样方便训练时能用同一日期索引
"""

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='/home/daiab/log/quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)


class DataPreprocess:
    def __init__(self, time_step):
        self.time_step = time_step
        self.ZScore = ZScore()
        self.SoftmaxHandle = SoftmaxHandle()
        self.RateNorm = RateNorm()
        self.BuildSerial = BuildSerial(time_step)
        self.option = Option()

    """
    input: pandas DataFrame type, shape like [sample_size, variable_size]
    """
    def process(self, origin_data):
        option = self.option

        date_time = origin_data.index.values
        self.date_time_range = date_time_range = date_time[self.time_step:-1]

        if option.train_data_norm_type == "zscore":
            self.train_data = \
                self.BuildSerial.generate_train_serial(origin_data, norm_type="zscore")[date_time_range]
        elif option.train_data_norm_type == "rate":
            self.train_data = \
                self.BuildSerial.generate_train_serial(origin_data, norm_type="rate")[date_time_range]
        else:
            raise Exception("norm data type error")

        origin_target_data = origin_data.loc[:, option.predict_index_type]

        """notification: here shifted by -1"""
        if option.target_data_norm_type == "zscore":
            self.target = self.ZScore.norm_to_zscore(origin_target_data).shift(-1).loc[date_time_range]
        elif option.target_data_norm_type == "rate":
            self.target = self.RateNorm.norm_to_rate(origin_target_data).shift(-1).loc[date_time_range]
        else:
            raise Exception("norm data type error")

        # self.rate = self.RateNorm.norm_to_rate(origin_target_data).shift(-1).loc[date_time_range]
        self.softmax = self.SoftmaxHandle.generate_softmax_target(origin_target_data).shift(-1).loc[date_time_range]
        self.days = self.target.shape[0]

    def index_to_date_time(self, index):
        return self.date_time_range[index]




class ZScore:
    def __init__(self):
        pass

    """
    input: pandas DataFrame type data, shape like [sample_size, variable_size]
    output: [sample_size, variable_size], but the output is shifted by -1, so the last row is [NaN, ..., NaN]
    """
    def norm_to_zscore(self, data_frame):
        return (data_frame - data_frame.mean(axis=0)) / data_frame.std(axis=0)



class SoftmaxHandle:
    def __init__(self):
        pass

    """
    input: pandas DataFrame type, shape like [sample_size, 1], just handle one variable per time
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
    def generate_softmax_target(self, data_frame):
        sample_size = data_frame.shape[0]
        softmax = pd.DataFrame(np.zeros(shape=[sample_size, 2], dtype=float),
                               index=data_frame.index, columns=['yes', 'no'])
        for row in range(1, sample_size):
            if data_frame.iat[row] >= data_frame.iat[row - 1]:
                softmax.iloc[row] = [1, 0]
            else:
                softmax.iloc[row] = [0, 1]
        assert softmax.shape[0] == sample_size
        return softmax



class RateNorm:
    def __init__(self):
        pass

    """
    input: pandas DataFrame type, shape like [sample_size, variable_size]
            the first sample will handled to be [1, 1, ..., 1]
    output: [sample_size, variable_size]
    """
    def norm_to_rate(self, data_frame):
        sample_size = data_frame.shape[0]
        rate = data_frame.copy()
        for row in range(sample_size - 1, 0, -1):
            row_rate = data_frame.iloc[row] / data_frame.iloc[row - 1] - 1
            rate.iloc[row] = row_rate
        assert rate.shape[0] == sample_size
        return rate


class BuildSerial:
    """
    length: how many samples per serial
    """
    def __init__(self, length):
        self.length = length

    """
    input: pandas DataFrame type, shape like [sample_size, variable_size]
            the input will be normalized to z-score for preprocess
            norm_type = "zscore" or "rate"
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
    notification: 使用索引查询时，比如查询'2014-01-20', 时间跨度是5天，
    那么返回的是'2014-01-16至'2014-01-20'的数据
    """
    def generate_train_serial(self, data_frame, norm_type="zscore"):
        sample_size = data_frame.shape[0]
        if norm_type == "zscore":
            norm_data = ZScore().norm_to_zscore(data_frame)
        elif norm_type == "rate":
            norm_data = RateNorm().norm_to_rate(data_frame)
        else:
            raise Exception("norm_type is not found")

        date_time_range = data_frame.index.values

        dict_data_frame = {}
        reindex_to_number = list(range(self.length))
        for row in range(0, sample_size - self.length + 1):
            step_data = norm_data.iloc[row:self.length+row]
            step_data.reset_index(reindex_to_number, inplace=True, drop=True)
            correspond_date = date_time_range[self.length + row - 1]
            dict_data_frame[correspond_date] = step_data

        panel = pd.Panel.from_dict(dict_data_frame)

        assert panel.shape[0] == sample_size - self.length + 1
        return panel


if __name__ == "__main__":
    datahandle = DataPreprocess(2)
    datahandle.option.predict_index_type = "openPrice"

