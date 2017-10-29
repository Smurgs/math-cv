import os
import time
import tensorflow as tf

import mathcv.model
import mathcv.data_handler
from mathcv.config import config


class Trainer:

    def __init__(self):
        self._sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self._dev_list = self._build_device_list()
        self._data_loader = mathcv.data_handler.DataLoader()
        self._graph_created = False

    def _save_graph(self):
        if not os.path.exists(config['saver_path']):
            os.makedirs(config['saver_path'])
        saver = tf.train.Saver()
        saver.save(tf.get_default_session(), config['saver_path'] + 'model')

    def _prepare_graph(self):
        if self._graph_created is False:
            print ('Checking for saved checkpoint')
            if os.path.isfile(config['saver_path'] + 'model.meta'):
                print ('Restoring model')
                saver = tf.train.import_meta_graph(config['saver_path'] + 'model.meta')
                saver.restore(tf.get_default_session(), config['saver_path'] + 'model')
            else:
                print ('Building model')
                self._build_graph()
                self._sess.run(tf.global_variables_initializer())
            self._graph_created = True

    def _build_graph(self):
        grads = []
        optimizer = tf.train.AdadeltaOptimizer(config['learning_rate'])
        global_step = tf.train.create_global_step()
        for x in range(len(self._dev_list)):
            with tf.name_scope('model%d' % x) as scope:
                with tf.device(self._dev_list[x]):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True if x > 0 else None):
                        model = mathcv.model.Model(self._data_loader.get_vocab_size())
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
        if config['num_gpus'] == 0:
            device_list.append('/cpu:0')
        elif config['num_gpus'] > 0:
            device_list += ['/gpu:%d' % x for x in range(config['num_gpus'])]
        else:
            raise ValueError('num_gpus is not defined or is not valid')
        return device_list

    def _feed_data(self, iter):
        img_placeholders = tf.get_collection(mathcv.model.Model.IMG_PLACEHOLDER_COLLECTION)
        input_placeholders = tf.get_collection(mathcv.model.Model.INPUT_PLACEHOLDER_COLLECTION)
        target_placeholders = tf.get_collection(mathcv.model.Model.TARGET_PLACEHOLDER_COLLECTION)
        assert len(img_placeholders) == len(input_placeholders) == len(target_placeholders)

        feeder_dict = {}
        for x in range(len(img_placeholders)):
            img, inp, target = iter.get_next()
            feeder_dict[img_placeholders[x]] = img
            feeder_dict[input_placeholders[x]] = inp
            feeder_dict[target_placeholders[x]] = target

        return feeder_dict

    def _get_average_batch_accuracy(self, train=True):
        if train:
            accs = tf.get_collection(mathcv.model.Model.TRAIN_ACCURACY_COLLECTION)
        else:
            accs = tf.get_collection(mathcv.model.Model.INFERENCE_ACCURACY_COLLECTION)

        expanded_accs = []
        for acc in accs:
            expanded_acc = tf.expand_dims(acc, 0)
            expanded_accs.append(expanded_acc)

        acc = tf.concat(axis=0, values=expanded_accs)
        acc = tf.reduce_mean(acc, 0)
        return acc

    def train(self):
        self._prepare_graph()
        merged_summaries = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config['summary_path'], self._sess.graph)

        print ('Starting training at global step: %s' % tf.train.global_step(self._sess, tf.train.get_global_step()))
        epoch_start_time = time.time()
        iter = self._data_loader.get_train_iterator()
        try:
            while True:
                feeder_dict = self._feed_data(iter)
                apply_grad_op = tf.get_default_graph().get_tensor_by_name('apply_grad_op:0')
                accuracy = self._get_average_batch_accuracy()
                _, acc, summary = self._sess.run([apply_grad_op, accuracy, merged_summaries], feed_dict=feeder_dict)
                summary_writer.add_summary(summary, tf.train.global_step(self._sess, tf.train.get_global_step()))
                print ('Batch accuracy: ' + str(acc))
        except EOFError:
            print ("Time for epoch:%f mins" % ((time.time() - epoch_start_time) / 60))

        self.validate()
        exit(1)
        self._save_graph()

    def validate(self):
        self._prepare_graph()
        print ('Running on validation set')
        iter = self._data_loader.get_validation_iterator()
        accs = []
        try:
            while True:
                feeder_dict = self._feed_data(iter)
                accuracy = self._get_average_batch_accuracy(train=False)
                val_acc = self._sess.run(accuracy, feed_dict=feeder_dict)
                accs.append(val_acc)
        except EOFError:
            val_acc = self._sess.run(tf.reduce_mean(accs))
            print ('Validation accuracy: ' + str(val_acc))


if __name__ == '__main__':
    t = Trainer()
    t.train()
