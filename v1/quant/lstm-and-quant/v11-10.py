import datetime

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
        self.is_back_test = True
        # 多长时间跨度作为上下文, 10 maybe better
        self.TIME_STEP = 20
        # lstm的隐藏单元个数, 5 maybe better
        self.NUM_HIDDEN = 20
        # 迭代次数 20 maybe better
        self.epochs = 1
        # 测试样本数
        if self.is_back_test:
            self.test_sample_num = 40
        else:
            self.test_sample_num = 40
        # whether plot datasource
        self.is_plot_line = True
        self._session = tf.Session()
        # self.init_plot()

    def get_data(self):
        data = ts.get_hist_data('000001')
        result, origin_data = self.format_data_dim(data)
        self.origin_data = origin_data
        self.all_data_num = len(result)
        self.all_data = result
        # print("result is >>>>>>>>>>>>>>>>>>")
        # print(result)


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
        # print("reverse_data datasource==========>")
        # print(datasource)
        return data

    def concatenate_data(self, data):
        data = np.concatenate(
            [data.open.values[:, np.newaxis],
             data.close.values[:, np.newaxis],
             data.high.values[:, np.newaxis],
             data.low.values[:, np.newaxis]], 1)
        # print("concat datasource==========>")
        # print(datasource)
        return data

    def normalization(self, data):
        rows = data.shape[0]
        minus = np.concatenate([[data[0, :]], data[0:rows - 1, :]], 0)
        # print("normalization datasource==========>")
        # print(datasource / minus - 1)
        return data / minus - 1

    def build_sample(self, data):
        result = []
        rows = data.shape[0]
        for i in range(self.TIME_STEP, rows + 1):
            tmp = []
            tmp.append(data[i - self.TIME_STEP: i, :])
            result.append(np.array(tmp))
        return result

    def get_one_epoch_data(self, index):
        # print("get_one_epoch_data >>>>>>>>>>>>>>")
        # print(self.all_data[index])
        return self.all_data[index]

    def get_one_epoch_target(self, index):
        tmp = self.origin_data[self.TIME_STEP + index]
        target = []
        target.append(tmp)
        # print(np.array(target))
        # print("get_one_epoch_target >>>>>>>>>>>>>>")
        # print(np.array(target))
        return np.array(target)

    def build_graph(self):
        self.train_data = tf.placeholder(tf.float32, [1, self.TIME_STEP, 4])
        # self.train_target = tf.placeholder(tf.float32, [1, 4])
        self.real_target = tf.placeholder(tf.float32, [1, 4])
        cell = tf.nn.rnn_cell.LSTMCell(self.NUM_HIDDEN)
        val, state = tf.nn.dynamic_rnn(cell, self.train_data, dtype=tf.float32)
        self.val = tf.transpose(val, [1, 0, 2])
        self.last_time = tf.gather(self.val, self.TIME_STEP - 1)

        self.weight = tf.Variable(tf.truncated_normal([self.NUM_HIDDEN, 4], dtype=tf.float32))
        self.bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1, 4]))
        self.predict_price = tf.matmul(self.last_time, self.weight) + self.bias

        self.diff = tf.sqrt(tf.reduce_sum(tf.square(self.predict_price - self.real_target)))

        self.minimize = tf.train.AdamOptimizer().minimize(self.diff)

    def train(self):
        init_op = tf.initialize_all_variables()
        self._session.run(init_op)
        for i in range(self.epochs):
            print("epoch %i " % i)
            for day in range(self.all_data_num - self.test_sample_num):
                self._session.run(self.minimize,
                                  {self.train_data: self.get_one_epoch_data(day),
                                   self.real_target: self.get_one_epoch_target(day)})

                if day % 100 == 0:
                    predict_price = self._session.run(self.predict_price,
                                      {self.train_data: self.get_one_epoch_data(day),
                                       self.real_target: self.get_one_epoch_target(day)})

                    diff = self._session.run(self.diff,
                                      {self.train_data: self.get_one_epoch_data(day),
                                       self.real_target: self.get_one_epoch_target(day)})
                    # last_time = self._session.run(self.last_time,
                    #                          {self.train_data: self.get_one_epoch_data(day),
                    #                           self.real_target: self.get_one_epoch_target(day)})
                    # val = self._session.run(self.val,
                    #                               {self.train_data: self.get_one_epoch_data(day),
                    #                                self.real_target: self.get_one_epoch_target(day)})
                    print("------------------")
                    # print(val.shape)
                    # print(last_time)
                    print(predict_price)
                    print(self.get_one_epoch_target(day))
                    print(diff)

    def predict(self):
        predict = np.array([[0, 0, 0, 0]])
        real = np.array([[0, 0, 0, 0]])
        # from_index = self.all_data_num - self.test_sample_num
        from_index = 1
        day = [from_index - 1]
        for i in range(from_index, self.all_data_num - 1):
            predict_price = self._session.run(self.predict_price,
                                                  {self.train_data: self.get_one_epoch_data(i),
                                                   self.real_target: self.get_one_epoch_target(i)})
            real_price = self.get_one_epoch_target(i)
            # print(predict_price)
            predict = np.concatenate([predict, predict_price], 0)
            # print(predict)
            real = np.concatenate([real, real_price], 0)
            # print(real)
            day.append(i)
        # print(predict[:, 0])

        self.plot_line(day[1:], predict[1:, :], real[1:, :])

    def predict_some_day(self, data):
        predict_price = self._session.run(self.predict_price,
                                          {self.train_data: data})
        return predict_price

    def get_data_by_date(self, from_date, to_date):
        """date format 'yyyy-mm-dd"""
        data = ts.get_hist_data('000001', start=from_date, end=to_date)
        #result, _ = self.format_data_dim(datasource)
        return self.get_data_for_predict(data)

    def get_data_for_predict(self, data):
        data = self.reverse_data(data)
        data = self.concatenate_data(data)
        data = self.normalization(data)
        return self.build_one_sample(data)

    def build_one_sample(self, data):
        return [data[-self.TIME_STEP:]]


    def plot_line(self, days, predict, real):
        plt.plot(days, predict[:, 0], 'r-')
        plt.plot(days, real[:, 0], 'b-')
        plt.show()
        plt.plot(days, predict[:, 1], 'r-')
        plt.plot(days, real[:, 1], 'b-')
        plt.show()
        plt.plot(days, predict[:, 2], 'r-')
        plt.plot(days, real[:, 2], 'b-')
        plt.show()
        plt.plot(days, predict[:, 3], 'r-')
        plt.plot(days, real[:, 3], 'b-')
        plt.show()

    def run(self):
        self.get_data()
        self.build_graph()
        self.train()
        if not self.is_back_test:
            self.predict()

    def temp(self, from_date, to_date):
        p = ts.get_hist_data('000001', from_date, to_date)
        p = p.open.values
        if p.shape[0] != 2:
            return 0
        return p[0] / p[1] - 1


# =====================================================
# rqalpha run -f v11-10.py -s 2016-10-01 -e 2016-011-07 -o result.pkl --plot

def init(context):
    context.s1 = "000001.XSHE"
    context.lstm_model = LstmModel()
    context.TIME_STEP = context.lstm_model.TIME_STEP
    context.lstm_model.run()
    context.total_stock_num = 0



def handle_bar(context, bar_dict):
    print("predict price today is %f" % context.price)
    # num_share = how_many(context.price)
    if context.price > 0:
        num_share = how_many(context.price)
        order_shares(context.s1, num_share)
        print("buy in ..................%d" % num_share)
        context.total_stock_num = context.total_stock_num + num_share
    else:
        # num_share = how_many(context.price)
        order_shares(context.s1, -context.total_stock_num)
        print("sold out ..................%d" % -context.total_stock_num)
        context.total_stock_num = 0


def before_trading(context, bar_dict):

    start_datetime = context.now - datetime.timedelta(days=context.TIME_STEP - 1) - datetime.timedelta(days=30)
    start_date = start_datetime.strftime('%Y-%m-%d')
    end_datetime = context.now  # - datetime.timedelta(days=1)
    end_date = end_datetime.strftime('%Y-%m-%d')
    tomorrow_datetime = context.now + datetime.timedelta(days=1)
    tomorrow_date = tomorrow_datetime.strftime('%Y-%m-%d')
    print("start date is %s, end date is %s" % (start_date, end_date))
    print("today is %s" % context.now)
    print("tomorrow price is %f" % context.lstm_model.temp(end_date, tomorrow_date))

    data = context.lstm_model.get_data_by_date(start_date, end_date)
    context.price = context.lstm_model.predict_some_day(data)[0][0]

def how_many(price):
    num_share = price * 500000
    if num_share < 100:
        num_share = 100
    elif num_share > 5000:
        num_share = 5000
    return num_share

