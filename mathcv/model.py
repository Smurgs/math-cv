import tensorflow as tf
import functools


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
    def __init__(self, images, labels, params):
        self.images = images
        self.labels = labels
        self.params = params
        self.prediction
        self.optimize
        self.accuracy

    @lazy_property
    def prediction(self):
        assert 'label_length' in self.params
        assert 'decoder_memory_dim' in self.params
        assert 'vocab_size' in self.params
        assert 'dropout_prob' in self.params
        assert self.images is not None
        # Convolution layers
        conv1 = self.build_convolution_layer(self.images, [5, 5, 1, 32], 1, self.params['dropout_prob'], tf.nn.relu)
        conv2 = self.build_convolution_layer(conv1, [5, 5, 32, 64], 2, self.params['dropout_prob'], tf.nn.relu)
        conv3 = self.build_convolution_layer(conv2, [5, 5, 64, 128], 1, self.params['dropout_prob'], tf.nn.relu)
        conv4 = self.build_convolution_layer(conv3, [5, 5, 128, 256], 2, self.params['dropout_prob'], tf.nn.relu)
        conv5 = self.build_convolution_layer(conv4, [5, 5, 256, 384], 1, self.params['dropout_prob'], tf.nn.relu)

        # Encoder - Fully connected layer
        fc_out = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(conv5), self.params['label_length'])

        # Decoder
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.params['decoder_memory_dim'], state_is_tuple=True)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(lstm_fw_cell, tf.expand_dims(fc_out, -1), dtype=tf.float32)
        return tf.contrib.layers.fully_connected(rnn_outputs, self.params['vocab_size'])

    @lazy_property
    def optimize(self):
        assert 'learning_rate' in self.params
        assert self.labels is not None
        assert self.prediction is not None
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.labels))
        return tf.train.AdadeltaOptimizer(self.params['learning_rate']).minimize(cross_entropy)

    @lazy_property
    def accuracy(self):
        assert self.labels is not None
        assert self.prediction is not None
        int_predictions = tf.argmax(self.prediction, axis=2)
        return tf.reduce_mean(tf.cast(tf.equal(self.labels, int_predictions), tf.float32))

    @staticmethod
    def build_convolution_layer(input_feed, shape, k, dropout, activation):
        weights = tf.Variable(tf.random_normal(shape, stddev=0.1))
        bias = tf.Variable(tf.zeros([shape[3]]))
        conv = tf.nn.bias_add(tf.nn.conv2d(input_feed, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
        conv = activation(conv)
        pool = tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        return tf.nn.dropout(pool, dropout)