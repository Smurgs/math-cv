import os
import sys
import pickle
import logging
import argparse
import itertools
import collections

import numpy as np


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
        padding = (200 - len(tokens)) * "\\<eos> "
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
    assert len(label) == 200
    skip_window = 1
    index = 0
    target = skip_window
    span = 2 * skip_window + 1
    batch_size = skip_window * 2 * (len(label) - skip_window * 2)
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
    if not os.path.exists('target'):
        os.makedirs('target')
    with open(parameters.mapper_path, 'wb') as m:
        pickle.dump(dictionary, m)

    # Transform labels
    logging.info('Transforming labels...')
    transformed_labels = transform_labels(padded_formulas, dictionary)

    logging.info('Sample batch...')
    batch_data, batch_labels = generate_batch(transformed_labels[0])
    print transformed_labels[0]
    print batch_data[:10]
    print "-----"
    print batch_labels[:10]


if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
