"""
使用收盘价训练模型
训练时，使用过去一段时间的历史数据作为输入，然后将lstm隐藏层的每个单元的输出
结果（向量）与同shape的权值向量相乘，然后结果与真实价格两个标量作均方误差，训练模型
"""

import tensorflow as tf
import tushare as ts
import numpy as np
import collections
import matplotlib.pyplot as plt

class LstmModel:
    def __init__(self):
        # 多长时间跨度作为上下文, 10 maybe better
        self.TIME_STEP = 15
        # lstm的隐藏单元个数, 5 maybe better
        self.NUM_HIDDEN = 15
        # 迭代次数 20 maybe better
        self.epochs = 50
        # 目前训练样本到什么位置了
        self.index = self.TIME_STEP - 1
        # 测试样本数
        self.test_sample_num = 40
        # whether print log
        self.log_print_on = True
        # whether plot datasource
        self.plot_figure_on = False
        self.is_plot_line = True

        self.buffer = collections.deque(maxlen=self.TIME_STEP)
        self._session = tf.Session()
        self.init_plot()

    def init_plot(self):
        if self.is_plot_line:
            pass
        else:
            plt.ion()
            plt.axis([0, 600, 6, 15])

    def gen_data(self):
        if self.index >= self.series_length:
            print("error")
        if self.index <= self.TIME_STEP - 1:
            for single in self.stock_example[:self.TIME_STEP]:
                self.buffer.append(single)
        else:
            self.buffer.append(self.stock_example[self.index])

        result = []
        result.append(list(self.buffer))
        return result

    def gen_target(self):
        # return np.array(datasource[index])
        # print(datasource[index + 1][np.newaxis, :])
        return self.stock_example[self.index + 1][np.newaxis, :]

    def get_data(self):
        stock = ts.get_hist_data("000001")
        close_price = stock.close.values
        # print(close_price)
        self.series_length = len(close_price)
        self.stock_example = close_price[:, np.newaxis]
        # print(self.stock_example[:40])

    def build_graph(self):
        self.data = tf.placeholder(tf.float32, [1, self.TIME_STEP, 1])
        self.target = tf.placeholder(tf.float32, [1, 1])

        # cell = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN)
        cell = tf.nn.rnn_cell.LSTMCell(self.NUM_HIDDEN)
        val, state = tf.nn.dynamic_rnn(cell, self.data, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        self.lstm_output = tf.gather(val, self.TIME_STEP - 1)
        # self.lstm_output = tf.gather(self.lstm_output, 0)
        # self.lstm_output = tf.gather(self.lstm_output, self.NUM_HIDDEN - 1)


        # weight = tf.Variable(tf.truncated_normal(shape=[self.NUM_HIDDEN, 1]), dtype=tf.float32)
        # bias = tf.Variable(tf.constant(10.0, shape=[1, 1]), dtype=tf.float32)
        self.weight = tf.Variable(tf.truncated_normal([self.NUM_HIDDEN, 1]), dtype=tf.float32)
        self.bias = tf.Variable(tf.constant(10.0, shape=[1]), dtype=tf.float32)

        self.stock_predict_price = tf.matmul(self.lstm_output, self.weight) + self.bias

        self.diff = tf.reduce_sum(np.square(self.stock_predict_price - self.target))
        # diff = np.square(lstm_output - target)
        self.minimize = tf.train.AdamOptimizer().minimize(self.diff)

    def train(self):
        init_op = tf.initialize_all_variables()
        self._session.run(init_op)
        for i in range(self.epochs):
            print("epoch %i " % i)
            for day in range(self.TIME_STEP - 1, self.series_length - self.test_sample_num - 1, 1):
                self.index = day
                self._session.run(self.minimize, {self.data: self.gen_data(), self.target: self.gen_target()})
                predict_stock_price = self._session.run(
                    self.stock_predict_price, {self.data: self.gen_data(), self.target: self.gen_target()})
                diff_price = self._session.run(self.diff, {self.data: self.gen_data(), self.target: self.gen_target()})

                if self.index % 100 == 0 and self.log_print_on:
                    print("epoch is %d ,stock price is %f ,and real price is %f, diff is %f"
                          % (i, predict_stock_price, self.gen_target()[0][0], diff_price))

                if i >= self.epochs - 1 and self.plot_figure_on:
                    self.plot_scatter(day, predict_stock_price, self.gen_target()[0][0])

        # self._session.close()

    def plot_scatter(self, no_day, predict, real):
        plt.scatter(no_day, predict, c='r')
        plt.scatter(no_day, real, c='b')
        plt.pause(0.000000001)

    def plot_line(self, days, predict, real):
        plt.plot(days, predict, 'r-')
        plt.plot(days, real, 'b-')
        plt.show()

    def run(self):
        self.get_data()
        self.build_graph()
        self.train()

    def test(self):
        test_index = self.series_length - self.test_sample_num
        predict, real, days = [], [], []
        tmp_index = 1
        for day in range(test_index, self.series_length - 1, 1):
            self.index = day
            predict_stock_price = self._session.run(
                self.stock_predict_price, {self.data: self.gen_data()})[0][0]
            real_stock_price = self.gen_target()[0][0]
            print("test===> predict price is %f, real price is %f" % (predict_stock_price, real_stock_price))
            days.append(tmp_index)
            predict.append(predict_stock_price)
            real.append(real_stock_price)
            tmp_index += 1
        self.plot_line(days, predict, real)

    # 进行预测时先把数据进行处理成合适的维度，input：1Dim output：3Dim
    def handle_data_format(self, data):
        data = data[:, np.newaxis]
        tmp = []
        for i in data:
            tmp.append(i)
        result = []
        result.append(tmp)
        print(result)
        return result

    # 用训练好的模型进行预测, datasource: 1Dim
    def predict(self, data):
        data = self.handle_data_format(data)
        predict_stock_price = self._session.run(
            self.stock_predict_price, {self.data: data})
        print("predict price is %f" % predict_stock_price)
        return predict_stock_price


    def destory(self):
        self._session.close()


if __name__ == "__main__":
    lstm = LstmModel()
    lstm.run()
    lstm.test()
    lstm.destory()
