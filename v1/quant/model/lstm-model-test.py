"""
1.使用open，close， high， low四个变量作为输入，target也是这四个变量
2.先对数据进行归一化，然后在进行训练
3.训练数据，先预先处理好，不再使用队列
"""
import numpy as np
import tushare as ts
import tensorflow as tf
import matplotlib.pyplot as plt

class LstmModel:
    def __init__(self):
        self.is_back_test = False
        # 多长时间跨度作为上下文, 10 maybe better
        self.TIME_STEP = 20
        # lstm的隐藏单元个数, 5 maybe better
        self.NUM_HIDDEN = 20
        # 迭代次数 20 maybe better
        self.epochs = 1
        # 目前训练样本到什么位置了
        self.index = self.TIME_STEP - 1
        # 测试样本数
        if self.is_back_test:
            self.test_sample_num = 0
        else:
            self.test_sample_num = 40
        # whether plot datasource
        self.is_plot_line = True
        # self.buffer = collections.deque(maxlen=self.TIME_STEP)
        self._session = tf.Session()
        # self.init_plot()

    def get_data(self):
        data = ts.get_hist_data('000001', '2016-11-01', '2016-11-07')
        print("tushare csv_data==========>")
        print(data)
        result, origin_data = self.format_data_dim(data)
        self.origin_data = origin_data
        self.all_data_num = len(result)
        self.all_data = result
        print("result csv_data==========>")
        print(result)



    def format_data_dim(self, data):
        data = self.reverse_data(data)
        data = self.concatenate_data(data)
        # rows = datasource.shape[0]
        origin_data = self.normalization(data)
        self.origin_data = origin_data
        data = self.build_sample(origin_data)
        return data, origin_data

    def reverse_data(self, data):
        data = data.reindex(index=data.index[::-1])
        print("reverse_data csv_data==========>")
        print(data)
        return data

    def concatenate_data(self, data):
        data = np.concatenate(
            [data.open.values[:, np.newaxis],
             data.close.values[:, np.newaxis],
             data.high.values[:, np.newaxis],
             data.low.values[:, np.newaxis]], 1)
        print("concat csv_data==========>")
        print(data)
        return data

    def normalization(self, data):
        rows = data.shape[0]
        minus = np.concatenate([[data[0, :]], data[0:rows - 1, :]], 0)
        print("normalization csv_data==========>")
        print(data / minus - 1)
        return data / minus - 1

    def build_sample(self, data):
        result = []
        rows = data.shape[0]
        for i in range(2, rows + 1):
            tmp = []
            tmp.append(data[i - 2: i, :])
            result.append(np.array(tmp))
        return result

lstm = LstmModel()
lstm.get_data()
