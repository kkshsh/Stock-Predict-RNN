import numpy as np
import tensorflow as tf

input = np.arange(20, dtype=float).reshape(4, 5)
target = np.array([5, 4, 3, 3], dtype=float)

def buildGraph():
    trainData = tf.placeholder(tf.float32, shape=[4, 5])
    targetData = tf.placeholder(tf.float32, shape=[4])
    weight = tf.Variable(tf.truncated_normal([5, 1], dtype=tf.float32))
    bias = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[1]))

    predict = tf.nn.xw_plus_b(x=trainData, weights=weight, biases=bias)
    minimizer = tf.train.AdamOptimizer().minimize(tf.reduce_sum(tf.squared_difference(predict, targetData)))
    clip = tf.clip_by_value(predict, clip_value_min=-15, clip_value_max=1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(1):
            dict = {trainData: input, targetData: target}
            sess.run(minimizer, feed_dict=dict)
            p = sess.run(predict, feed_dict=dict)
            print(p)
            clip = sess.run(clip, feed_dict=dict)
            print(clip)

buildGraph()




