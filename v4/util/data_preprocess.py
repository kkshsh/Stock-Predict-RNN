import logging
import numpy as np
from v4.config import config
from v4.entry.train_data_struct import DD

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

shape_0 = 0


class DataPreprocess:
    def __init__(self, time_step):
        self.time_step = time_step

    def process(self, origin_data, date_range, operate_type):
        """
        input origin_date: numpy array type, shape like [sample_size, variable_size]
        """
        global shape_0
        shape_0 = origin_data.shape[0]

        """generate_train_serial 返回的数据长度是(sample_size - time_step + 1)"""
        train_data = generate_train_serial(origin_data, self.time_step, norm_type=config.train_data_norm_type)

        """generate target data 返回的数据长度是(sample_size)"""
        origin_target_data = origin_data[:, config.predict_index_type]
        target = norm_data(origin_target_data, config.target_data_norm_type)

        """generate softmax data 返回的数据长度是(sample_size)"""
        softmax = generate_softmax_target(origin_target_data)

        if operate_type == config.OFFLINE_TRAIN:
            split_data_to_train_test = True
        elif operate_type == config.ONLINE_TRAIN or operate_type == config.PREDICT:
            split_data_to_train_test = False
        else:
            raise Exception("type is error")

        if operate_type == config.OFFLINE_TRAIN or operate_type == config.ONLINE_TRAIN:
            """
            因为最开始第一天的数据，如果是使用rate norm的方式，第一天的数据是有问题的，所以这里把他排除出去
            warning：然后为了统一，在线训练时也把第一天数据给去掉了
            """
            filter_date_range = date_range[self.time_step - 1 + 1: shape_0 - 1]
            if target is not None:
                return DD(filter_date_range, train_data=train_data[1:-1], softmax=softmax[self.time_step + 1:],
                          target=target[self.time_step + 1:], is_split_data_to_train_test=split_data_to_train_test)
            else:
                return DD(filter_date_range, train_data=train_data[1:-1], softmax=softmax[self.time_step + 1:],
                          is_split_data_to_train_test=split_data_to_train_test)

        elif operate_type == config.PREDICT:
            filter_date_range = date_range[self.time_step - 1: shape_0]
            assert len(filter_date_range) == train_data.shape[0]
            return DD(filter_date_range, train_data=train_data, is_split_data_to_train_test=split_data_to_train_test)

        else:
            raise Exception("type is error")





def norm_data(origin_target_data, norm_type):
    if norm_type == "zscore":
        return norm_to_zscore(origin_target_data)
    elif norm_type == "rate":
        return norm_to_rate(origin_target_data)
    elif norm_type == "none":
        return None
    else:
        raise Exception("norm data type error")


def norm_to_zscore(origin_data):
    """
    input: pandas DataFrame type data, shape like [sample_size, variable_size]
    output: [sample_size, variable_size], but the output is shifted by -1, so the last row is [NaN, ..., NaN]
    """
    return (origin_data - origin_data.mean(axis=0)) / origin_data.std(axis=0)


def norm_to_rate(orgin_data):
    """
    input: pandas DataFrame type, shape like [sample_size, variable_size]
            the first sample will handled to be [1, 1, ..., 1]
    output: [sample_size, variable_size]
    """
    rate = np.ones_like(orgin_data)
    for row in range(shape_0 - 1, 0, -1):
        row_rate = orgin_data[row] / orgin_data[row - 1] - 1
        rate[row] = row_rate
    return rate


def generate_softmax_target(origin_data):
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
    warn: 注意append的顺序
    """
    softmax = [[0, 0]]
    for row in range(1, shape_0):
        if origin_data[row] >= origin_data[row - 1]:
            softmax.append([1, 0])
        else:
            softmax.append([0, 1])
    assert len(softmax) == shape_0
    return np.array(softmax)


def generate_train_serial(origin_data, time_step, norm_type):
    """
    input: pandas DataFrame type, shape like [sample_size, variable_size]
            the input will be normalized to z-score for preprocess
            norm_type = "zscore" or "rate"
         time_step: how many samples per serial
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
    if norm_type == "zscore":
        norm_data = norm_to_zscore(origin_data)
    elif norm_type == "rate":
        norm_data = norm_to_rate(origin_data)
    else:
        raise Exception("norm_type is not found")

    train_data = []
    for row in range(time_step, shape_0 + 1):
        step_data = norm_data[row - time_step: row]
        train_data.append(step_data)

    assert len(train_data) == shape_0 - time_step + 1
    return np.array(train_data)


if __name__ == "__main__":
    datahandle = DataPreprocess(2)
