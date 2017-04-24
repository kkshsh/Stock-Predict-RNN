import time
import tensorflow as tf
import logging
from v5 import config as cfg
from v5.model import read_rec
import numpy as np
# from tensorflow.contrib.rnn import LayerNormBasicLSTMCell, BasicRNNCell, GridLSTMCell, GRUCell, BasicLSTMCell
from v5.model.bnlstm import BNLSTMCell
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%b %d %Y %H:%M:%S')


class LstmModel:
    def __init__(self, session):
        self.session = session
        self.right = 0
        self.samples = 0
        self.right_list = np.zeros([5])
        self.samples_list = np.zeros([5])
        self.w = {'fc_weight_1': tf.Variable(tf.truncated_normal([cfg.time_step * cfg.state_size, cfg.class_num], stddev=0.01, dtype=tf.float32), name='fc_weight_1'),
                  'fc_weight_2': tf.Variable(tf.truncated_normal([128, cfg.class_num], stddev=0.01, dtype=tf.float32), name='fc_weight_2')}
        self.b = {'fc_bias_1': tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[cfg.class_num]), name='fc_bias_1'),
                  'fc_bias_2': tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[cfg.class_num]), name='fc_bias_2')}

    def build_graph(self):
        """placeholder: train data"""
        self.batch_data, self.batch_label = read_rec.read_and_decode(cfg.rec_file)
        self.rnn_keep_prop = cfg.rnn_keep_prop

        # multi_cell = tf.contrib.rnn.MultiRNNCell([BasicLSTMCell(cfg.state_size)] * cfg.hidden_layers)
        # cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=self.rnn_keep_prop)
        multi_cell = tf.contrib.rnn.MultiRNNCell(
            [BNLSTMCell(cfg.state_size, training=True) for _ in range(cfg.hidden_layers)])
        state_init = multi_cell.zero_state(cfg.batch_size, dtype=tf.float32)
        val, self.states = tf.nn.dynamic_rnn(multi_cell, self.batch_data, initial_state=state_init, dtype=tf.float32)

        """reshape the RNN output"""
        # val = tf.transpose(val, [1, 0, 2])
        # self.val = tf.gather(val, val.get_shape()[0] - 1)
        dim = cfg.time_step * cfg.state_size
        self.val = tf.reshape(val, [-1, dim])


        self.logits = tf.nn.xw_plus_b(self.val, self.w['fc_weight_1'], self.b['fc_bias_1'])
        # self.logits = tf.nn.relu(tf.nn.xw_plus_b(self.val, self.w['fc_weight_1'], self.b['fc_bias_1']), name="relu")
        # fc_1 = tf.nn.dropout(fc_1, keep_prob=0.8, name="softmax_dropout")

        # self.logits = tf.nn.xw_plus_b(fc_1, self.w['fc_weight_2'], self.b['fc_bias_2'], name='fc_2')

        self.cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.batch_label, name="cross_entropy"))

        """Loss function and Optimizer"""
        global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64,
                               initializer=tf.zeros_initializer(),
                               trainable=False)
        self.poly_decay_lr = tf.train.polynomial_decay(learning_rate=cfg.learning_rate,
                                                  global_step=global_step,
                                                  decay_steps=cfg.decay_steps,
                                                  end_learning_rate=0.0002,
                                                  power=cfg.power)
        weight = [v for _, v in self.w.items()]
        norm = tf.add_n([tf.nn.l2_loss(i) for i in weight])
        # self.minimize = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate).\
        #     minimize(self.cross_entropy + cfg.weght_decay * norm, global_step=global_step)
        self.minimize = tf.train.MomentumOptimizer(
            learning_rate=self.poly_decay_lr, momentum=cfg.momentum).\
            minimize(self.cross_entropy + cfg.weght_decay * norm, global_step=global_step)

        """saver"""
        self.saver = tf.train.Saver()

    def save_model(self):
        save_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.saver.save(self.session, "log/%s.ckpt" % save_time)
        logging.info("save file time: %s", save_time)

    def train_model(self):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
        for i in range(cfg.iter_num):
            _, logits, labels = self.session.run([self.minimize, self.logits, self.batch_label])
            self.acc(logits, labels, i)
            if (i + 1) % 100 == 0:
                ce, lr = self.session.run([self.cross_entropy, self.poly_decay_lr])
                logging.info("%d th iter, cross_entropy == %s, learning rate == %s", i, ce, lr)
                logging.info('accuracy == %s',  self.right / self.samples)
                self.right = self.samples = 0

            # self.save_model()
        coord.request_stop()
        coord.join(threads=threads)

    def acc(self, logits, label, gs):
        max_idx = np.argmax(logits, axis=1)
        # if gs % 20 == 0:
        #     print(np.count_nonzero(max_idx == 1))
        #     print(np.count_nonzero(label == 1))
        #     print('--------------')
        equal = np.sum(np.equal(max_idx, label).astype(int))
        self.right += equal
        self.samples += cfg.batch_size

    def acc_dist(self, logits, labels):
        threshold = np.arange(0.5, 1, 0.1)
        max = np.max(logits, axis=1, keepdims=True)
        prob = np.exp(logits - max) / np.sum(np.exp(logits - max), axis=1, keepdims=True)
        b_labels = labels.astype(bool)[:, np.newaxis]
        b_idx = np.concatenate((~b_labels, b_labels), axis=1)
        target = b_idx.astype(int)
        for i, t in enumerate(threshold):
            bool_index = prob > t
            self.samples_list[i] += np.count_nonzero(bool_index)
            self.right_list[i] += np.count_nonzero(target[bool_index])


def run():
    with tf.Graph().as_default(), tf.Session() as session:
        lstmModel = LstmModel(session)
        lstmModel.build_graph()
        """init variables"""
        if cfg.ckpt_file is None:
            logging.info("init all variables by random")
            lstmModel.session.run(tf.global_variables_initializer())
        else:
            logging.info("init all variables by previous file")
            lstmModel.saver.restore(lstmModel.session, cfg.ckpt_file)
        lstmModel.train_model()


if __name__ == '__main__':
    run()
