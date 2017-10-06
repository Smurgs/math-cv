import os
import time
import numpy as np
import tensorflow as tf

from mathcv.model import Model
from mathcv.config import config
from mathcv.data_loader import DataLoader

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('vocab_size', 563, 'vocabulary size')
tf.app.flags.DEFINE_integer('learning_rate', 0.1, 'learning rate')
tf.app.flags.DEFINE_integer('decoder_memory_dim', 512, 'decoder lstm memory size')
tf.app.flags.DEFINE_float('dropout_prob', 0.75, 'dropout probability')



#
#     print "Training"
#     for i in range(epochs):
#         if reduce_lr == 5:
#             lr = max(min_lr, lr - 0.005)
#             reduce_lr = 0
#         print "Epoch %d learning rate %.4f" % (i, lr)
#         epoch_start_time = time.time()
#         batch_50_start = epoch_start_time
#         for j in range(len(train)):
#             images, labels = train[j]
#
#             print "Starting batch #: " + str(j)
#             train_step.run(feed_dict={learning_rate: lr, data_input: images, true_labels: labels})
#             print "Batch accuracy: " + str(accuracy.eval(feed_dict={data_input: images, true_labels: labels}))
#         print "Time for epoch:%f mins" % ((time.time() - epoch_start_time) / 60)
#         print "Running on Validation Set"
#         accs = []
#         for j in range(len(val)):
#             images, labels = val[j]
#             val_accuracy = accuracy.eval(feed_dict={data_input: images,
#                                                     true_labels: labels})
#             accs.append(val_accuracy)
#         val_acc = sess.run(tf.reduce_mean(accs))
#         if (val_acc - last_val_acc) >= .01:
#             reduce_lr = 0
#         else:
#             reduce_lr = reduce_lr + 1
#         last_val_acc = val_acc
#         print("val accuracy %g" % val_acc)
#
#     print 'Saving model'
#     saver = tf.train.Saver()
#     id = 'model-' + time.strftime("%d-%m-%Y--%H-%M")
#     os.mkdir(id)
#     save_path = saver.save(sess, id + '/model')
#     print 'Running on Test Set'
#     accs = []
#     for j in range(len(test)):
#         images, labels = test[j]
#         test_accuracy = accuracy.eval(feed_dict={data_input: images,
#                                                  true_labels: labels})
#         accs.append(test_accuracy)
#     test_acc = sess.run(tf.reduce_mean(accs))
#     print("test accuracy %g" % test_acc)


def main(argv):
    print 'Starting mathcv'
    train_model()


def preprocess():
    pass


def train_model():
    print ('Loading data')
    data_loader = DataLoader()

    print ('Building model')
    image_input = tf.placeholder(tf.float32, [None, config['image_height'], config['image_width'], 1])
    label_input = tf.placeholder(tf.int64, [None, config['label_length']])
    hp = {
        'label_length': config['label_length'],
        'decoder_memory_dim': FLAGS.decoder_memory_dim,
        'vocab_size': FLAGS.vocab_size,
        'dropout_prob': FLAGS.dropout_prob,
        'learning_rate': FLAGS.learning_rate
    }
    model = Model(image_input, label_input, hp)

    epochs = 1

    print ('Starting session')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("log", sess.graph)

        print ('Starting training')
        for epoch in range(epochs):
            print ("Starting epoch " + str(epoch + 1))
            epoch_start_time = time.time()
            for batch in data_loader.get_train_batches():
                images, labels = batch
                _, acc = sess.run([model.optimize, model.accuracy], feed_dict={image_input: images, label_input: labels})
                print ('Batch accuracy: ' + str(acc))
            print "Time for epoch:%f mins" % ((time.time() - epoch_start_time) / 60)

        print ('Running on validation set')
        accs = []
        for batch in data_loader.get_validation_batches():
            images, labels = batch
            val_acc = sess.run(model.accuracy, feed_dict={image_input: images, label_input: labels})
            accs.append(val_acc)
        val_acc = sess.run(tf.reduce_mean(accs))
        print ('Validation accuracy: ' + str(val_acc))


def infer():
    pass


if __name__ == '__main__':
    tf.app.run()