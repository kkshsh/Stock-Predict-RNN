time_step = 40
is_training = True
batch_size = 1200
rec_file = '/home/daiab/machine_disk/code/Stock-Predict-RNN/v5/data/c3.tfrecords'
state_size = 200
hidden_layers = 5
rnn_keep_prop = 1.0
num_samples = 1e7
epochs = 10
iter_num = int(num_samples / batch_size * 20)
learning_rate = 0.008
momentum = 0.9
power = 0.6
weght_decay = 0.0002
decay_steps = iter_num
ckpt_file = None
class_num = 3
