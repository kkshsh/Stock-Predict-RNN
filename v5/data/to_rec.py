import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import os
import random

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%b %d %Y %H:%M:%S')

FIELDS = ['ticker', 'tradeDate', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice', 'isOpen']
PRICE_IDX = [2, 3, 4, 5]
REF=3  # close price
TIME_STEP = 40
SHUFFLE = None
SHUFFLE_STEP = 1


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='read csv file')
    parser.add_argument('csv_fold', help='csv fold path which contains all csv files', type=str)
    parser.add_argument('--shuffle', default=1, help='csv fold path which contains all csv files', type=int)
    args = parser.parse_args()
    return args


def process_data(pd_data):
    pd_data = pd_data[pd_data['isOpen'] == 1]
    pd_data.dropna(axis=0)
    np_data = pd_data[FIELDS].as_matrix()
    data = np_data[:, PRICE_IDX]
    ratio = data[1:] / data[:-1] - 1
    ret_data = []
    ret_label = []
    days = ratio.shape[0]
    logging.info('total days : {}'.format(days))
    for i in range(TIME_STEP, days, 1):
        ret_data.append(ratio[i - TIME_STEP: i])
        r = ratio[i, REF]
        # 0 down, 2 up, others 1
        ret_label.append(2 if r > 0.01 else (0 if r < -0.01 else 1))
    assert len(ret_data) == len(ret_label), 'data and label mismatch'
    return ret_data, ret_label


def to_tfrecord(writer, data, label):
    row = data.shape[0]
    assert row == label.shape[0], 'to_tfrecord errors'
    if SHUFFLE:
        idx = list(range(row))
        random.shuffle(idx)
        data = data[idx]
        label = label[idx]
    for i in range(row):
        example = tf.train.Example(features=tf.train.Features(feature={
                'label' : int64_feature(label[i]),
                'data': float_feature(data[i].ravel())})) # need to reshape to one dim
        writer.write(example.SerializeToString())


def read_csv(args):
    writer = tf.python_io.TFRecordWriter('_.tfrecords')
    data = []
    label = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(args.csv_fold):
        for file in filenames:
            pd_data = pd.read_csv('{}/{}'.format(dirpath, file), sep=',')
            d, l = process_data(pd_data)
            data.extend(d)
            label.extend(l)
            if (count + 1) % SHUFFLE_STEP == 0:
                to_tfrecord(writer, np.array(data).astype(np.float), np.array(label).astype(np.int))
                data=[]
                label=[]
                logging.info('{} th write to rec'.format(count))
            count += 1
    if len(data) != 0:
        to_tfrecord(writer, np.array(data).astype(np.float), np.array(label).astype(np.int))
    writer.close()


def main():
    args = parse_args()
    SHUFFLE = True if args.shuffle == 1 else False
    logging.info('shuffle = {}'.format(SHUFFLE))
    read_csv(args)
    pass


if __name__ == '__main__':
    main()
