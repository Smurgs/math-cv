import tensorflow as tf
import functools

from mathcv.config import config


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Model:
    model_count = 0

    def __init__(self, images, labels, vocab_size):
        Model.model_count += 1
        self.images = images
        self.labels = labels
        self.vocab_size = vocab_size
        with tf.name_scope('model%d' % Model.model_count) as scope:
            self.prediction
            self.loss
            self.accuracy
            tf.get_variable_scope().reuse_variables()

    @lazy_property
    def prediction(self):
        # Convolution layers
        conv1 = self.build_convolution_layer(self.images, [5, 5, 1, 32], 1, 'conv_layer_1')
        conv2 = self.build_convolution_layer(conv1, [5, 5, 32, 64], 2, 'conv_layer_2')
        conv3 = self.build_convolution_layer(conv2, [5, 5, 64, 128], 1, 'conv_layer_3')
        conv4 = self.build_convolution_layer(conv3, [5, 5, 128, 256], 2, 'conv_layer_4')
        conv5 = self.build_convolution_layer(conv4, [5, 5, 256, 384], 1, 'conv_layer_5')

        # Encoder - Fully connected layer
        with tf.name_scope('encoder') as scope:
            fc_out = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(conv5), config['label_length'], scope='shared/encoder/fcl')

        # Decoder
        with tf.name_scope('decoder') as scope:
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(config['decoder_memory_dim'], state_is_tuple=True)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(lstm_fw_cell, tf.expand_dims(fc_out, -1), dtype=tf.float32, scope='shared/decoder/rnn')
            out = tf.contrib.layers.fully_connected(rnn_outputs, self.vocab_size, scope='shared/decoder/fcl')
        return out

    @lazy_property
    def loss(self):
        assert self.labels is not None
        assert self.prediction is not None
        with tf.name_scope('loss') as scope:
            out = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.labels))
        return out

    @lazy_property
    def accuracy(self):
        assert self.labels is not None
        assert self.prediction is not None
        with tf.name_scope('accuracy') as scope:
            int_predictions = tf.argmax(self.prediction, axis=2)
            out = tf.reduce_mean(tf.cast(tf.equal(self.labels, int_predictions), tf.float32))
        return out

    @staticmethod
    def build_convolution_layer(input_feed, shape, k, scope_name, dropout=config['dropout_prob'], activation=tf.nn.relu):
        with tf.name_scope(scope_name) as scope:
            weights = tf.Variable(tf.random_normal(shape, stddev=0.1))
            bias = tf.Variable(tf.zeros([shape[3]]))
            conv = tf.nn.bias_add(tf.nn.conv2d(input_feed, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
            conv = activation(conv)
            pool = tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
            out = tf.nn.dropout(pool, dropout)
        return out
