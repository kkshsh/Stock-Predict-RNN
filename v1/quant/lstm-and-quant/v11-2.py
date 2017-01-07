import tushare as ts
import datetime
import numpy as np
import tensorflow as tf
import tushare as ts
import collections
import matplotlib.pyplot as plt

class LstmModel:
    def __init__(self):
        # 多长时间跨度作为上下文, 10 maybe better
        self.TIME_STEP = 10
        # lstm的隐藏单元个数, 5 maybe better
        self.NUM_HIDDEN = 5
        # 迭代次数 20 maybe better
        self.epochs = 30
        # 目前训练样本到什么位置了
        self.index = self.TIME_STEP - 1
        # 测试样本数
        self.test_sample_num = 30
        # whether print log
        self.log_print_on = False
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
        self.lstm_output = tf.gather(self.lstm_output, 0)
        self.lstm_output = tf.gather(self.lstm_output, self.NUM_HIDDEN - 1)


        # weight = tf.Variable(tf.truncated_normal(shape=[self.NUM_HIDDEN, 1]), dtype=tf.float32)
        # bias = tf.Variable(tf.constant(10.0, shape=[1, 1]), dtype=tf.float32)
        self.weight = tf.Variable(tf.truncated_normal([1]), dtype=tf.float32)
        self.bias = tf.Variable(tf.constant(10.0, shape=[1]), dtype=tf.float32)

        self.stock_predict_price = tf.mul(self.lstm_output, self.weight) + self.bias

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
                self.stock_predict_price, {self.data: self.gen_data()})
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
        # print(result)
        return result

    # 用训练好的模型进行预测, datasource: 1Dim
    def predict(self, data):
        print(data)
        data = self.handle_data_format(data)
        predict_stock_price = self._session.run(
            self.stock_predict_price, {self.data: data})
        print("predict price is %f" % predict_stock_price)
        return predict_stock_price


    def destory(self):
        self._session.close()

# =====================================================
#使用预测的当天的数据，比较昨天真实的数据，决定是否买卖
# rqalpha run -f v11-2.py -s 2016-10-01 -e 2016-011-03 -o result.pkl --plot


def init(context):
    context.s1 = "000001.XSHE"
    context.lstm_model = LstmModel()
    context.TIME_STEP = context.lstm_model.TIME_STEP
    context.lstm_model.run()



def handle_bar(context, bar_dict):
    print("last day price is %f and predict today price is %f" %(context.data[-1], context.predict_price))
    if context.predict_price + 2 > context.data[-1]:
        order_shares(context.s1, 101)
        print("buy in")
    elif context.predict_price + 2 < context.data[-1]:
        order_shares(context.s1, -101)
        print("sold out")


def before_trading(context, bar_dict):
    # 由于存在周末空档的情况，所以先多取一段时间(10days)的数据回来，然后在下面的处理中只取最近日期的部分条数据
    start_datetime = context.now - datetime.timedelta(days=context.TIME_STEP - 1) - datetime.timedelta(days=20)
    start_date = start_datetime.strftime('%Y-%m-%d')
    now_date = context.now.strftime('%Y-%m-%d')
    print("start date and now date is ===> (%s, %s)" % (start_date, now_date))
    # 这里只截取了最后TIME STEP时间的数据
    context.data_yesterday = ts.get_hist_data(context.s1.split(".")[0], start=start_date, end=now_date).close.values[-(context.TIME_STEP):]
    context.data_yesterday_yesterday = ts.get_hist_data(context.s1.split(".")[0], start=start_date, end=now_date).close.values[-(context.TIME_STEP + 1): -1]
    # context.csv_data = handle_data_format(np.array(csv_data))
    context.predict_price_today = context.lstm_model.predict(context.data_yesterday)
    context.predict_price_yesterday = context.lstm_model.predict(context.data_yesterday_yesterday)

