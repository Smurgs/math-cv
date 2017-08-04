import os
import sys
import math
import pickle
import logging
import argparse
import itertools
import collections

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def process_args(args):
    parser = argparse.ArgumentParser(description='Train word embeddings for tokenized formulas')

    parser.add_argument('--data-path', dest='data_path',
                        type=str, required=True,
                        help=('Input file path containing <img_path> <label_idx> per line.'
                        ))
    parser.add_argument('--output-path', dest='output_path',
                        type=str, required=True,
                        help=('Output file path containing <img_path> <label_idx> per line.'
                        ))
    parser.add_argument('--label-path', dest='label_path',
                        type=str, required=True,
                        help=('Input label path containing <tokenized formula> per line.'
                        ))
    parser.add_argument('--mapper-output', dest='mapper_path',
                        type=str, default='target/mapper.txt',
                        help=('Output path for dictionary that maps tokens to integers.'
                        ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt'
                        ))
    parameters = parser.parse_args(args)
    return parameters


def exit_if_path_invalid(path):
    if not os.path.exists(path):
        logging.warning('%s does not exist!' % os.path.basename(path))
        exit(1)
    return path


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def load_data(label_path, data_path):
    tokenized_formulas = [x[:-1] for x in open(label_path).readlines()]
    padded_formulas = []
    for _, formula_index in [x[:-1].split() for x in open(data_path).readlines()]:
        tokens = tokenized_formulas[int(formula_index) - 1].split()
        padding = (label_length - len(tokens)) * "\\<eos> "
        padded_formulas.append(tokens + padding.split())
    return padded_formulas


def build_dictionary(padded_formulas):
    combined_list = list(itertools.chain.from_iterable(padded_formulas))
    counts = list(collections.Counter(combined_list).most_common(5000))
    dictionary = dict()
    for word, _ in counts:
        dictionary[word] = len(dictionary)
    dictionary['\\<unkn>'] = len(dictionary)
    return dictionary


def transform_labels(padded_formulas, dictionary):
    transformed_labels = []
    for padded_formula in padded_formulas:
        transformed_label = []
        for token in padded_formula:
            if token in dictionary:
                transformed_label.append(dictionary[token])
            else:
                logging.warning('Failed to find token "' + token + '" in dictionary')
                transformed_label.append(dictionary['\\<unkn>'])
        transformed_labels.append(transformed_label)
    return transformed_labels


def generate_batch(label):
    assert len(label) == label_length
    index = 0
    target = skip_window
    span = 2 * skip_window + 1
    data = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(label[index])
        index += 1
    for i in range(0, batch_size, skip_window*2):
        for j in range(skip_window):
            data[i + j] = buffer[target]
            labels[i + j] = buffer[j]
        for j in range(skip_window+1, span):
            data[i + j - 1] = buffer[target]
            labels[i + j - 1] = buffer[j]
        buffer.append(label[index])
        index = (index + 1) % len(label)

    return data, labels


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)


# Hyper parameters
num_steps = 1000
valid_size = 16
valid_window = 100
skip_window = 1
num_sampled = 64
label_length = 200
embedding_size = 128
batch_size = skip_window * 2 * (label_length - skip_window * 2)
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


def main(args):
    parameters = process_args(args)
    setup_logging(parameters.log_path)

    logging.info('Script being executed: %s' % __file__)
    data_path = exit_if_path_invalid(parameters.data_path)
    label_path = exit_if_path_invalid(parameters.label_path)

    # Load data
    logging.info('Loading data...')
    padded_formulas = load_data(label_path, data_path)

    # Build vocabulary
    logging.info('Building vocabulary...')
    dictionary = build_dictionary(padded_formulas)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    vocabulary_size = len(dictionary)
    if not os.path.exists('target'):
        os.makedirs('target')
    with open(parameters.mapper_path, 'wb') as m:
        pickle.dump(dictionary, m)

    # Transform labels
    logging.info('Transforming labels...')
    transformed_labels = transform_labels(padded_formulas, dictionary)

    # Build graph
    logging.info('Building graph...')
    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device('/cpu:0'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(transformed_labels[step])
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 100 == 0:
                if step > 0:
                    average_loss /= 100
                # The average loss is an estimate of the loss over the last 100 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

    print 'All done!'

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
