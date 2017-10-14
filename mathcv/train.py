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


def train():
    print ('Loading data')
    data_loader = mathcv.data_handler.DataLoader()

    print ('Building model')
    image_input = tf.placeholder(tf.float32, [None, config['image_height'], config['image_width'], 1], name='image_input')
    label_input = tf.placeholder(tf.int64, [None, config['label_length']], name='label_input')
    model = mathcv.model.Model(image_input, label_input, data_loader.get_vocab_size())
    opt = tf.train.AdadeltaOptimizer(config['learning_rate'])
    grads_and_vars = opt.compute_gradients(model.loss)
    apply_grad_op = opt.apply_gradients(grads_and_vars)

    print ('Starting session')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("mathcv/target/model_summaries", sess.graph)

        print ('Starting training')
        for epoch in range(config['epochs']):
            print ("Starting epoch " + str(epoch + 1))
            epoch_start_time = time.time()
            for batch in data_loader.get_train_batches()[:5]:
                images, labels = batch
                _, acc = sess.run([apply_grad_op, model.accuracy], feed_dict={image_input: images, label_input: labels})
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