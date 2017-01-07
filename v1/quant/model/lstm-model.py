import tensorflow as tf
import tushare as ts
import numpy as np
import collections
import matplotlib.pyplot as plt

# 多长时间跨度作为上下文
TIME_STEP = 10
# lstm的隐藏单元个数
NUM_HIDDEN = 23
# 迭代次数
epochs = 0
# 目前训练样本到什么位置了
index = TIME_STEP - 1
# is first iterator
is_first = True
# predict array
predict_arr = []

buffer = collections.deque(maxlen=TIME_STEP)



def gen_data(data):
    global index
    global is_first
    global series_length
    global buffer
    if index >= series_length:
        print("error")
    if is_first:
        for single in data[:TIME_STEP]:
            buffer.append(single)
        is_first = False
    else:
        buffer.append(data[index])

    result = []
    result.append(list(buffer))
    return result

def gen_target(data):
    global index
    # return np.array(datasource[index])
    # print(datasource[index + 1][np.newaxis, :])
    return data[index + 1][np.newaxis, :]


plt.ion()
plt.axis([0, 600, 6, 15])
def plot(i, predict, real):
    plt.scatter(i, predict, c='r')
    plt.scatter(i, real, c='b')
    plt.pause(0.000000001)

stock = ts.get_hist_data("000001")
close_price = stock.close.values
# print(close_price)
stock_example = close_price[:, np.newaxis]
# print(stock_example)

series_length = len(close_price)


data = tf.placeholder(tf.float32, [1, TIME_STEP, 1])
target = tf.placeholder(tf.float32, [1, 1])

# cell = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN)
cell = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN)
val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
lstm_output = tf.gather(val, TIME_STEP - 1)

weight = tf.Variable(tf.truncated_normal(shape=[1, 1]), dtype=tf.float32)
bias = tf.Variable(tf.constant(10.0, shape=[1, 1]), dtype=tf.float32)
stock_predict_price = tf.mul(lstm_output, weight) + bias


diff = tf.reduce_sum(np.square(stock_predict_price - target))
# diff = np.square(lstm_output - target)
minimize = tf.train.AdamOptimizer().minimize(diff)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    epochs = 10
    for i in range(epochs):
        for j in range(TIME_STEP, series_length - 1, 1):
            index = j
            sess.run(minimize, {data: gen_data(stock_example), target: gen_target(stock_example)})
            stock_price = sess.run(stock_predict_price, {data: gen_data(stock_example), target: gen_target(stock_example)})
            diff_price = sess.run(diff, {data: gen_data(stock_example), target: gen_target(stock_example)})
            if index % 50 == 0:
                print("epoch is %d ,stock price is %f ,and real price is %f, diff is %f"
                      % (i, stock_price[0][0], gen_target(stock_example)[0][0], diff_price))
                # predict_arr.append(stock_price)
            if i >= epochs - 1:
                plot(j, stock_price[0][0], gen_target(stock_example)[0][0])

