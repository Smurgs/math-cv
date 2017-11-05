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

    TRAIN_ACCURACY_COLLECTION = "train_accuracy_collection"
    CLONE_TRAIN_ACCURACY_COLLECTION = "clone_train_accuracy_collection"
    INFERENCE_ACCURACY_COLLECTION = "inference_accuracy_collection"
    CLONE_INFERENCE_ACCURACY_COLLECTION = "clone_inference_accuracy_collection"
    IMG_PLACEHOLDER_COLLECTION = "img_placeholder_collection"
    INPUT_PLACEHOLDER_COLLECTION = "input_placeholder_collection"
    TARGET_PLACEHOLDER_COLLECTION = "target_placeholder_collection"

    def __init__(self, vocab_size):
        # Vocabulary size
        self.vocab_size = vocab_size

        # Images placeholder
        self.images = tf.placeholder(tf.float32, [None, config['image_height'], config['image_width'], 1], name='image_input')
        tf.add_to_collection(Model.IMG_PLACEHOLDER_COLLECTION, self.images)

        # Training input label placeholder
        self.label_inputs = tf.placeholder(tf.int64, [None, config['label_length']], name='label_input')
        tf.add_to_collection(Model.INPUT_PLACEHOLDER_COLLECTION, self.label_inputs)

        # Target label placeholder
        self.label_targets = tf.placeholder(tf.int64, [None, config['label_length']], name='label_target')
        tf.add_to_collection(Model.TARGET_PLACEHOLDER_COLLECTION, self.label_targets)

        # Model components
        self._encoder
        self.decoder
        self.loss
        self.accuracy

    @lazy_property
    def _encoder(self):
        with tf.variable_scope('encoder') as scope:
            conv1 = self._build_convolution_layer(self.images, [5, 5, 1, 32], 1, 'conv_layer_1')
            conv2 = self._build_convolution_layer(conv1, [5, 5, 32, 64], 2, 'conv_layer_2')
            conv3 = self._build_convolution_layer(conv2, [5, 5, 64, 128], 1, 'conv_layer_3')
            conv4 = self._build_convolution_layer(conv3, [5, 5, 128, 256], 2, 'conv_layer_4')
            conv5 = self._build_convolution_layer(conv4, [5, 5, 256, 384], 1, 'conv_layer_5')
            flattened = tf.contrib.layers.flatten(conv5)
            with tf.device('/cpu:0'):
                tf.get_variable('fcl/weights', [flattened.get_shape()[1], config['embedding_size']], initializer=tf.random_normal_initializer())
                tf.get_variable('fcl/biases', config['embedding_size'], initializer=tf.constant_initializer(0))
            fc_out = tf.contrib.layers.fully_connected(inputs=tf.contrib.layers.flatten(conv5),
                                                       num_outputs=config['embedding_size'],
                                                       activation_fn=None,
                                                       scope='fcl', reuse=True)
        return fc_out

    @lazy_property
    def decoder(self):
        with tf.variable_scope('decoder') as scope:
            # Prepare lstm initial state with encoder data
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(config['decoder_memory_dim'], state_is_tuple=True)
            zero_state = lstm_cell.zero_state(batch_size=config['batch_size'], dtype=tf.float32)
            _, initial_state = lstm_cell(self._encoder, zero_state)

            # Setup embedding map
            with tf.device('/cpu:0'):
                embedding_map = tf.get_variable(name='map', shape=[self.vocab_size, config['embedding_size']])

            # Training decoder
            train_state_tuple = initial_state
            inp_embeddings = tf.nn.embedding_lookup(embedding_map, self.label_inputs)
            list_inputs = tf.split(inp_embeddings, config['label_length'], axis=1)
            output_ta = tf.TensorArray(size=config['label_length'], dtype=tf.float32)
            input_ta = tf.TensorArray(size=config['label_length'], dtype=tf.float32)
            input_ta = input_ta.unstack(list_inputs)

            def train_time_step(time_t, output_ta_t, state):
                in_value = input_ta.read(time_t)
                out_value, out_state = lstm_cell(tf.squeeze(in_value, axis=[1]), state)
                output_ta_t = output_ta_t.write(time_t, out_value)
                return time_t+1, output_ta_t, out_state

            time = tf.constant(0, dtype='int32')
            time_steps = config['label_length']
            _, output_final_ta, final_state = tf.while_loop(
                cond=lambda time, *_: time < time_steps,
                body=train_time_step,
                loop_vars=(time, output_ta, train_state_tuple),
                parallel_iterations=config['batch_size'],
                swap_memory=False)

            training_outputs = output_final_ta.stack()
            training_outputs = tf.transpose(training_outputs, perm=[1, 0, 2])
            training_outputs = tf.reshape(training_outputs,
                                          shape=[-1, config['decoder_memory_dim']])
            with tf.device('/cpu:0'):
                tf.get_variable('fcl/weights', [config['decoder_memory_dim'], self.vocab_size], initializer=tf.random_normal_initializer())
                tf.get_variable('fcl/biases', self.vocab_size, initializer=tf.constant_initializer(0))
            logits = tf.contrib.layers.fully_connected(inputs=training_outputs,
                                                       num_outputs=self.vocab_size,
                                                       activation_fn=None,
                                                       scope='fcl', reuse=True)

            # Validation decoder
            val_state_tuple = initial_state
            first_input = tf.slice(self.label_inputs, [0, 0], [-1, 1])
            inference_inp = tf.squeeze(tf.nn.embedding_lookup(embedding_map, first_input), axis=[1])
            output_ta = tf.TensorArray(size=config['label_length'], dtype=tf.int64)
            input_ta = tf.TensorArray(size=config['label_length']+1, dtype=tf.float32)
            input_ta = input_ta.write(0, inference_inp)

            def inference_time_step(time_t, input_ta_t, output_ta_t, state):
                in_value = input_ta_t.read(time_t)
                out_value, out_state = lstm_cell(in_value, state)
                inference_logits = tf.contrib.layers.fully_connected(inputs=out_value,
                                                                     num_outputs=self.vocab_size,
                                                                     activation_fn=None,
                                                                     scope='fcl',
                                                                     reuse=True)
                inferences_softmax = tf.nn.softmax(inference_logits, name="inference_softmax")
                inference_int_prediction = tf.argmax(inferences_softmax, axis=1)

                output_ta_t = output_ta_t.write(time_t, inference_int_prediction)
                input_ta_t = input_ta_t.write(time_t+1, tf.nn.embedding_lookup(embedding_map, inference_int_prediction))
                return time_t+1, input_ta_t, output_ta_t, out_state

            time = tf.constant(0, dtype='int32')
            time_steps = config['label_length']
            _, _, output_final_ta, final_state = tf.while_loop(
                cond=lambda time, *_: time < time_steps,
                body=inference_time_step,
                loop_vars=(time, input_ta, output_ta, val_state_tuple),
                parallel_iterations=config['batch_size'],
                swap_memory=False)

            inference_outputs = output_final_ta.stack()
            inference_outputs = tf.transpose(inference_outputs)
            inference_outputs = tf.reshape(inference_outputs, [-1])

        with tf.variable_scope('inference_accuracy') as scope:
            targets = tf.reshape(self.label_targets, [-1])
            inference_accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, inference_outputs), tf.float32))
            tf.add_to_collection(Model.CLONE_INFERENCE_ACCURACY_COLLECTION, inference_accuracy)
        return logits

    @lazy_property
    def loss(self):
        with tf.variable_scope('loss') as scope:
            targets = tf.reshape(self.label_targets, [-1])
            out = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.decoder, labels=targets))
            tf.summary.scalar('clone_loss', out)
            tf.add_to_collection(tf.GraphKeys.LOSSES, out)
        return out

    @lazy_property
    def accuracy(self):
        with tf.variable_scope('accuracy') as scope:
            int_predictions = tf.argmax(self.decoder, axis=1)
            targets = tf.reshape(self.label_targets, [-1])
            out = tf.reduce_mean(tf.cast(tf.equal(targets, int_predictions), tf.float32), name='accuracy')
            tf.add_to_collection(Model.CLONE_TRAIN_ACCURACY_COLLECTION, out)
            tf.summary.scalar('clone_accuracy', out)
        return out

    @staticmethod
    def average_batch_accuracy():
        with tf.variable_scope('avg_batch_acc'):
            with tf.device('/cpu:0'):
                accs = tf.get_collection(Model.CLONE_TRAIN_ACCURACY_COLLECTION)
                expanded_accs = []
                for acc in accs:
                    expanded_acc = tf.expand_dims(acc, 0)
                    expanded_accs.append(expanded_acc)

                acc = tf.concat(axis=0, values=expanded_accs)
                acc = tf.reduce_mean(acc, 0)
                tf.add_to_collection(Model.TRAIN_ACCURACY_COLLECTION, acc)

    @staticmethod
    def average_inference_batch_accuracy():
        with tf.variable_scope('avg_inference_batch_acc'):
            with tf.device('/cpu:0'):
                accs = tf.get_collection(Model.CLONE_INFERENCE_ACCURACY_COLLECTION)
                expanded_accs = []
                for acc in accs:
                    expanded_acc = tf.expand_dims(acc, 0)
                    expanded_accs.append(expanded_acc)

                acc = tf.concat(axis=0, values=expanded_accs)
                acc = tf.reduce_mean(acc, 0)
                tf.add_to_collection(Model.INFERENCE_ACCURACY_COLLECTION, acc)

    @staticmethod
    def _build_convolution_layer(input_feed, shape, k, scope_name, dropout=config['dropout_prob'], activation=tf.nn.relu):
        with tf.variable_scope(scope_name) as scope:
            with tf.device('/cpu:0'):
                weights = tf.get_variable('weights', shape, initializer=tf.random_normal_initializer())
                bias = tf.get_variable('bias', shape[3], initializer=tf.constant_initializer(0))
            conv = tf.nn.bias_add(tf.nn.conv2d(input_feed, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
            conv = activation(conv)
            pool = tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
            out = tf.nn.dropout(pool, dropout)
        return out
