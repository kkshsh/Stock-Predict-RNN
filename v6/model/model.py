import time
import tensorflow as tf
import logging
from v6 import config as cfg
from v6.model import read_rec
import numpy as np
# from tensorflow.contrib.rnn import LayerNormBasicLSTMCell, BasicRNNCell, GridLSTMCell, GRUCell, BasicLSTMCell
from v6.model.bnlstm import BNLSTMCell
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
        self.w = {'ce_weight':
                      tf.Variable(tf.truncated_normal([cfg.time_step * cfg.state_size, cfg.class_num], stddev=0.01,
                                                      dtype=tf.float32), name='ce_weight'),
                  'rg_weight':
                      tf.Variable(tf.truncated_normal([cfg.time_step * cfg.state_size, 1], stddev=0.01,
                                                      dtype=tf.float32), name='ce_weight'),
                  }
        self.b = {'ce_bias':
                      tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[cfg.class_num]), name='ce_bias'),
                  'rg_bias':
                      tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[1]), name='ce_bias')
                  }

    def build_graph(self):
        """placeholder: train data"""
        self.batch_data, self.batch_label, self.batch_target = read_rec.read_and_decode(cfg.rec_file)
        self.rnn_keep_prop = cfg.rnn_keep_prop

        # multi_cell = tf.contrib.rnn.MultiRNNCell([BasicLSTMCell(cfg.state_size)] * cfg.hidden_layers)
        # cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=self.rnn_keep_prop)
        multi_cell = tf.contrib.rnn.MultiRNNCell(
            [BNLSTMCell(cfg.state_size, training=True) for _ in range(cfg.hidden_layers)])
        state_init = multi_cell.zero_state(cfg.batch_size, dtype=tf.float32)
        val, self.states = tf.nn.dynamic_rnn(multi_cell, self.batch_data, initial_state=state_init, dtype=tf.float32)

        """reshape the RNN output"""
        dim = cfg.time_step * cfg.state_size
        self.val = tf.reshape(val, [-1, dim])


        self.logits = tf.nn.xw_plus_b(self.val, self.w['ce_weight'], self.b['ce_bias'])
        reg = tf.nn.xw_plus_b(self.val, self.w['rg_weight'], self.b['rg_weight'])
        self.reg_loss = tf.reduce_mean(tf.sqrt(tf.squared_difference(reg, self.batch_target)))

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
        loss = self.reg_loss + self.cross_entropy + cfg.weght_decay * norm
        self.minimize = tf.train.MomentumOptimizer(
            learning_rate=self.poly_decay_lr, momentum=cfg.momentum).\
            minimize(loss, global_step=global_step)

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
            self.acc(logits, labels)
            if (i + 1) % 20 == 0:
                ce, rl, lr = self.session.run([self.cross_entropy, self.reg_loss, self.poly_decay_lr])
                logging.info("%d th iter, cross_entropy == %s, reg loss = %s, learning rate == %s", i, ce, rl, lr)
                logging.info('accuracy == %s',  self.right / self.samples)
                self.right = 0
                self.samples = 0
            if (i + 1) % 10000 == 0:
                # self.save_model()
                pass

        coord.request_stop()
        coord.join(threads=threads)

    def acc(self, logits, label):
        max_idx = np.argmax(logits, axis=1)
        equal = np.sum(np.equal(max_idx, label).astype(int))
        self.right += equal
        self.samples += cfg.batch_size


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
