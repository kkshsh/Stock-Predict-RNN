import tensorflow as tf
import v5.config as cfg


def read_and_decode(record_file):
    print(record_file)
    # read_and_decode_test(record_file)
    data_queue = tf.train.input_producer([record_file], capacity=1e5, name="string_input_producer")
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(data_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'data': tf.FixedLenFeature([cfg.time_step * 4], tf.float32)})
    data_raw = features['data']
    label = features['label']
    data = tf.reshape(data_raw, [cfg.time_step, 4])
    data.set_shape([cfg.time_step, 4])
    if cfg.is_training:
        data_batch, label_batch = tf.train.batch([data, label],
                                                     batch_size=cfg.batch_size,
                                                     capacity=cfg.batch_size * 50,
                                                     num_threads=4)
        return data_batch, label_batch
    else:
        return tf.expand_dims(data, 0), tf.expand_dims(label, 0)


def read_and_decode_test(record_file):
    print('test......')
    record_iterator = tf.python_io.tf_record_iterator(path=record_file)
    print(record_iterator)
    for data in record_iterator:
        print('iter ....... ')
        example = tf.train.Example()
        example.ParseFromString(data)
        d = example.features.feature['data'].float_list.value[0:200]
        l = example.features.feature['label'].int64_list.value[0]
        print(len(d))
        print(l)
        print('------------------')
        break
