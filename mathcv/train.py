import os
import time
import tensorflow as tf

import mathcv.model
import mathcv.data_handler
from mathcv.config import config


class Trainer:
    def __init__(self):
        self._epochs = config['epochs']
        self._summary_path = config['summary_path']
        self._num_gpus = config['num_gpus']
        self._saver_path = config['saver_path']
        self._image_height = config['image_height']
        self._image_width = config['image_width']
        self._label_length = config['label_length']
        self._learning_rate = config['learning_rate']
        self._dev_list = self._build_device_list()
        self._data_loader = mathcv.data_handler.DataLoader()

    def _save_graph(self):
        if not os.path.exists(self._saver_path):
            os.makedirs(self._saver_path)
        saver = tf.train.Saver()
        saver.save(tf.get_default_session(), self._saver_path + 'model')

    def _restore_graph(self):
        saver = tf.train.import_meta_graph(self._saver_path + 'model.meta')
        saver.restore(tf.get_default_session(), self._saver_path + 'model')

    def _build_graph(self):
        grads = []
        optimizer = tf.train.AdadeltaOptimizer(self._learning_rate)
        global_step = tf.train.create_global_step()
        for x in range(len(self._dev_list)):
            image_input = tf.placeholder(tf.float32, [None, self._image_height, self._image_width, 1],
                                         name='image_input%d' % x)
            label_input = tf.placeholder(tf.int64, [None, self._label_length], name='label_input%d' % x)

            with tf.name_scope('model%d' % x) as scope:
                with tf.device(self._dev_list[x]):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True if x > 0 else None):
                        model = mathcv.model.Model(image_input, label_input, self._data_loader.get_vocab_size())
                        grad = optimizer.compute_gradients(model.loss)
                        grads.append(grad)

        with tf.device('/cpu:0'):
            avg_grad = self._average_gradients(grads)
            optimizer.apply_gradients(avg_grad, global_step=global_step, name='apply_grad_op')

    def _average_gradients(self, grads):
        with tf.name_scope('average_gradients') as scope:
            average_grads = []
            for grad_and_vars in zip(*grads):
                grads = []
                for g, _ in grad_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
        return average_grads

    def _build_device_list(self):
        device_list = []
        if self._num_gpus == 0:
            device_list.append('/cpu:0')
        elif self._num_gpus > 0:
            device_list += ['/gpu:%d' % x for x in range(self._num_gpus)]
        else:
            raise ValueError('num_gpus is not defined or is not valid')
        return device_list

    def train(self):
        print ('Loading data')
        data_loader = mathcv.data_handler.DataLoader()

        print ('Checking for saved checkpoint')
        checkpoint_exists = os.path.isfile(self._saver_path + 'model.meta')

        if not checkpoint_exists:
            print ('Building model')
            self._build_graph()

        print ('Starting session')
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            if checkpoint_exists:
                print ('Restoring model')
                self._restore_graph()
            else:
                sess.run(tf.global_variables_initializer())

            merged_summaries = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self._summary_path, sess.graph)

            print ('Starting training')
            for epoch in range(self._epochs):
                print ('Global step: %s' % tf.train.global_step(sess, tf.train.get_global_step()))
                print ("Starting epoch " + str(epoch + 1))
                epoch_start_time = time.time()
                train_batches = data_loader.get_train_batches()
                num_clones = len(self._dev_list)
                for x in range(0, int(len(train_batches)/num_clones)*num_clones, num_clones):
                    feeder = {}
                    for i in range(num_clones):
                        images, labels = train_batches[x+i]
                        feeder[tf.get_default_graph().get_tensor_by_name('image_input%d:0' % i)] = images
                        feeder[tf.get_default_graph().get_tensor_by_name('label_input%d:0' % i)] = labels
                        apply_grad_op = tf.get_default_graph().get_tensor_by_name('apply_grad_op:0')
                        accuracy = tf.get_default_graph().get_tensor_by_name('model0/accuracy/accuracy:0')
                    _, acc, summary = sess.run([apply_grad_op, accuracy, merged_summaries], feed_dict=feeder)
                    summary_writer.add_summary(summary, x)
                    print ('Batch accuracy: ' + str(acc))
                print ("Time for epoch:%f mins" % ((time.time() - epoch_start_time) / 60))
                self._save_graph()

            print ('Running on validation set')
            accs = []
            for batch in data_loader.get_validation_batches():
                images, labels = batch
                accuracy = tf.get_default_graph().get_tensor_by_name('model0/accuracy/accuracy:0')
                val_acc = sess.run(accuracy, feed_dict={
                    tf.get_default_graph().get_tensor_by_name('image_input0:0'): images,
                    tf.get_default_graph().get_tensor_by_name('label_input0:0'): labels
                })
                accs.append(val_acc)
            val_acc = sess.run(tf.reduce_mean(accs))
            print ('Validation accuracy: ' + str(val_acc))
