import time
import tensorflow as tf

import mathcv.model
import mathcv.data_handler
from mathcv.config import config


#     print 'Saving model'
#     saver = tf.train.Saver()
#     id = 'model-' + time.strftime("%d-%m-%Y--%H-%M")
#     os.mkdir(id)
#     save_path = saver.save(sess, id + '/model')

def average_gradients(grads):
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


def train():
    print ('Loading data')
    data_loader = mathcv.data_handler.DataLoader()

    print ('Building model')
    image_input = tf.placeholder(tf.float32, [None, config['image_height'], config['image_width'], 1], name='image_input')
    label_input = tf.placeholder(tf.int64, [None, config['label_length']], name='label_input')
    image_input2 = tf.placeholder(tf.float32, [None, config['image_height'], config['image_width'], 1], name='image_input')
    label_input2 = tf.placeholder(tf.int64, [None, config['label_length']], name='label_input')

    opt = tf.train.AdadeltaOptimizer(config['learning_rate'])
    with tf.device('/gpu:0'):
        model = mathcv.model.Model(image_input, label_input, data_loader.get_vocab_size())
        grad1 = opt.compute_gradients(model.loss)
    with tf.device('/gpu:1'):
        model2 = mathcv.model.Model(image_input2, label_input2, data_loader.get_vocab_size())
        grad2 = opt.compute_gradients(model2.loss)
    with tf.device('/cpu:0'):
        print ([grad1, grad2])
        grads = average_gradients([grad1, grad2])
        apply_grad_op = opt.apply_gradients(grads)

    print ('Starting session')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("mathcv/target/model_summaries", sess.graph)

        print ('Starting training')
        for epoch in range(config['epochs']):
            print ("Starting epoch " + str(epoch + 1))
            epoch_start_time = time.time()
            train_batches = data_loader.get_train_batches()[:4]
            for x in range(0, len(train_batches), 2):
                images, labels = train_batches[x]
                images2, labels2 = train_batches[x+1]
                _, acc = sess.run([apply_grad_op, model.accuracy], feed_dict={image_input: images, label_input: labels, image_input2:images2, label_input2:labels2})
                print ('Batch accuracy: ' + str(acc))
            print ("Time for epoch:%f mins" % ((time.time() - epoch_start_time) / 60))

        print ('Running on validation set')
        accs = []
        for batch in data_loader.get_validation_batches():
            images, labels = batch
            val_acc = sess.run(model.accuracy, feed_dict={image_input: images, label_input: labels})
            accs.append(val_acc)
        val_acc = sess.run(tf.reduce_mean(accs))
        print ('Validation accuracy: ' + str(val_acc))
