import os
import sys
import logging
import argparse
import itertools
import collections


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


def main(args):
    parameters = process_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s' % __file__)
    data_path = exit_if_path_invalid(parameters.data_path)
    output_path = exit_if_path_invalid(parameters.output_path)
    label_path = exit_if_path_invalid(parameters.label_path)

    # Load data
    logging.info('Loading data...')
    tokenized_formulas = open(label_path).readlines()
    padded_formulas = []
    for _, formula_index in open(data_path).readlines():
        tokens = tokenized_formulas[formula_index-1].split()
        padded_formulas.append(tokens + (200-len(tokens)) * "\\<eos>")

    # Build vocabulary
    logging.info('Building vocabulary...')
    combined_list = list(itertools.chain.from_iterable(padded_formulas))
    counts = list(collections.Counter(combined_list).most_common(5000))
    dictionary = dict()
    for word, _ in counts:
        dictionary[word] = len(dictionary)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')