import time
import numpy as np
import tensorflow as tf
from v4.config import config
from v4.util.load_all_code import load_all_code
from v4.util.mini_batch import batch
from v4.service.construct_train_data import GenTrainData

logger = config.get_logger(__name__)


class LstmModel:
    def __init__(self, session):
        self.session = session
        self.all_stock_code = [1] # load_all_code()
        self.loop_code_time = 0

    def load_data(self, operate_type, end_date=None, limit=None):
        """
        operate_type:
        case 1: 线下训练
        case 2: 在线训练
        case 3: 预测
        """
        self.dd_list = GenTrainData(self.all_stock_code, config.time_step, operate_type=operate_type, end_date=end_date,
                                    limit=limit).dd_list

    def build_graph(self):
        """placeholder: drop keep prop"""
        self.rnn_keep_prop = tf.placeholder(tf.float32)
        self.hidden_layer_keep_prop = tf.placeholder(tf.float32)

        """placeholder: train data"""
        self.one_train_data = tf.placeholder(tf.float32, [None, config.time_step, 4])
        self.one_target_data = tf.placeholder(tf.float32, [None, 2])

        """RNN architecture"""
        cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_cell_num,
                                            input_size=[config.batch_size, config.time_step, config.hidden_cell_num])
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=self.rnn_keep_prop)

        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.hidden_layer_num)
        val, self.states = tf.nn.dynamic_rnn(multi_cell, self.one_train_data, dtype=tf.float32)

        """reshape the RNN output"""
        # val = tf.transpose(val, [1, 0, 2])
        # self.val = tf.gather(val, val.get_shape()[0] - 1)
        dim = config.time_step * config.hidden_cell_num
        self.val = tf.reshape(val, [-1, dim])

        """softmax layer 1"""
        self.weight = tf.Variable(tf.truncated_normal([dim, config.output_cell_num], dtype=tf.float32), name='weight_1')
        self.bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1, config.output_cell_num]), name='bias_1')

        tmp_value = tf.nn.relu(tf.matmul(self.val, self.weight) + self.bias, name="relu")
        """softmax layer 1 drop out"""
        tmp_value = tf.nn.dropout(tmp_value, keep_prob=self.hidden_layer_keep_prop, name="softmax_dropout")

        """softmax layer 2"""
        self.weight_2 = tf.Variable(tf.truncated_normal([config.output_cell_num, 2], dtype=tf.float32), name='weight_2')
        self.bias_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1, 2]), name='bias_2')
        self.predict_target = tf.matmul(tmp_value, self.weight_2) + self.bias_2

        """Loss function and Optimizer"""
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.predict_target, self.one_target_data, name="cross_entropy"))
        self.minimize = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.cross_entropy)

        """saver"""
        self.saver = tf.train.Saver()

    def save_model(self):
        save_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.saver.save(self.session, "/home/daiab/ckpt/%s.ckpt" % save_time)
        logger.info("save file time: %s", save_time)

    def train_model(self, operate_type, epochs):
        for i in range(epochs):
            count = len(self.dd_list) - 1
            while count >= 0:
                dd = self.dd_list[count]

                logger.info("epoch == %d, count == %d, stockCode == %d", i, count, dd.code)
                batch_data = batch(batch_size=config.batch_size,
                                   data=dd.train_data[dd.train_index],
                                   softmax=dd.softmax[dd.train_index])
                feed_dict = {}
                for one_train_data, _, softmax in batch_data:
                    feed_dict = {self.one_train_data: one_train_data,
                                 self.one_target_data: softmax,
                                 self.rnn_keep_prop: config.rnn_keep_prop,
                                 self.hidden_layer_keep_prop: config.hidden_layer_keep_prop}
                    self.session.run(self.minimize, feed_dict=feed_dict)

                if (operate_type == config.OFFLINE_TRAIN) and len(feed_dict) != 0:
                    cross_entropy = self.session.run(self.cross_entropy, feed_dict=feed_dict)
                    logger.info("cross_entropy == %f", cross_entropy)
                    self.test_model(dd)
                count -= 1
            if config.is_save_file:
                self.save_model()

    def test_model(self, dd):
        count_arr, right_arr, prop_step_arr = np.zeros(6), np.zeros(6), np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        logger.info("test begin ......")
        train = dd.train_data[dd.test_index]
        softmax = dd.softmax[dd.test_index]

        predict = self.session.run(self.predict_target,
                                   feed_dict={self.one_train_data: train,
                                              self.rnn_keep_prop: 1.0,
                                              self.hidden_layer_keep_prop: 1.0})
        predict = np.exp(predict)
        probability = predict / np.sum(predict, axis=1)[:, np.newaxis]

        for i in range(6):
            prop_threshold = prop_step_arr[i]
            bool_index = probability > prop_threshold
            count_arr[i] = bool_index.sum()
            right_arr[i] = softmax[bool_index].sum()

        logger.info("test ratio>>%s", right_arr / count_arr)
        logger.info("test count>>%s", count_arr)


def run():
    # with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    with tf.Graph().as_default(), tf.Session() as session:
        config.config_print()
        lstmModel = LstmModel(session)
        lstmModel.build_graph()

        """init variables"""
        if config.init_variable_file_path == "":
            logger.info("init all variables by random")
            lstmModel.session.run(tf.initialize_all_variables())
        else:
            logger.info("init all variables by previous file")
            lstmModel.saver.restore(lstmModel.session, config.init_variable_file_path)

        # writer = tf.train.SummaryWriter("./model-summary", graph=tf.get_default_graph())
        # writer.close()

        lstmModel.load_data(operate_type=config.OFFLINE_TRAIN)
        lstmModel.train_model(operate_type=config.OFFLINE_TRAIN, epochs=config.offline_train_epochs)
        session.close()


if __name__ == '__main__':
    run()
