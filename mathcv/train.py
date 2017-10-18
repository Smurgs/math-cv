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

def _average_gradients(grads):
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


def _build_device_list():
    device_list = []
    if config['num_gpus'] == 0:
        device_list.append('/cpu:0')
    elif config['num_gpus'] > 0:
        device_list += ['/gpu:%d' % x for x in range(config['num_gpus'])]
    else:
        raise ValueError('num_gpus is not defined or is not valid')
    return device_list


def train():
    dev_list = _build_device_list()
    num_clones = len(dev_list)

    print ('Loading data')
    data_loader = mathcv.data_handler.DataLoader()

    print ('Building model')
    model_inputs = []
    grads = []
    optimizer = tf.train.AdadeltaOptimizer(config['learning_rate'])
    for x in range(num_clones):
        image_input = tf.placeholder(tf.float32, [None, config['image_height'], config['image_width'], 1], name='image_input')
        label_input = tf.placeholder(tf.int64, [None, config['label_length']], name='label_input')
        model_inputs.append((image_input, label_input))

        with tf.name_scope('model%d' % x) as scope:
            with tf.device(dev_list[x]):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True if x > 0 else None):
                    model = mathcv.model.Model(image_input, label_input, data_loader.get_vocab_size())
                    grad = optimizer.compute_gradients(model.loss)
                    grads.append(grad)

    with tf.device('/cpu:0'):
        avg_grad = _average_gradients(grads)
        apply_grad_op = optimizer.apply_gradients(avg_grad)

    merged_summaries = tf.summary.merge_all()

    print ('Starting session')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter("mathcv/target/model_summaries", sess.graph)

        print ('Starting training')
        for epoch in range(config['epochs']):
            print ("Starting epoch " + str(epoch + 1))
            epoch_start_time = time.time()
            train_batches = data_loader.get_train_batches()
            for x in range(0, int(len(train_batches)/num_clones)*num_clones, num_clones):
                feeder = {}
                for i in range(num_clones):
                    images, labels = train_batches[x+i]
                    feeder[model_inputs[i][0]] = images
                    feeder[model_inputs[i][1]] = labels
                _, acc, summary = sess.run([apply_grad_op, model.accuracy, merged_summaries], feed_dict=feeder)
                summary_writer.add_summary(summary, x)
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
