import numpy as np
from PIL import Image

from mathcv.config import config


class DataLoader:
    def __init__(self):
        self._train = None
        self._val = None
        self._test = None
        self._dictionary = None
        self.load_data()

    def get_train_batches(self):
        return self._train

    def get_validation_batches(self):
        return self._val

    def get_test_batches(self):
        return self._test

    def get_vocab_size(self):
        pass

    def load_data(self):
        formulas = open(config['formula_path']).read().split('\n')
        train = open(config['train_path']).read().split('\n')[:-1]
        val = open(config['val_path']).read().split('\n')[:-1]
        test = open(config['test_path']).read().split('\n')[:-1]

        train = combine_x_and_y(train[:10], formulas)
        val = combine_x_and_y(val[:10], formulas)
        test = combine_x_and_y(test[:10], formulas)

        dictionary = dict()
        dictionary['\\<start>'] = len(dictionary)
        dictionary['\\<end>'] = len(dictionary)
        dictionary['\\<pad>'] = len(dictionary)
        dictionary['\\<unknown>'] = len(dictionary)

        train = token_to_int(train, dictionary, True)
        val = token_to_int(val, dictionary)
        test = token_to_int(test, dictionary)

        self._train = batchify(train)
        self._val = batchify(val)
        self._test = batchify(test)
        self._dictionary = dictionary


def combine_x_and_y(lst, formulas):
    combined = []
    for y in lst:
        y = y.split(' ')
        img = np.array(Image.open(config['img_dir'] + y[0]).convert('L'))
        img = img[..., np.newaxis]
        combined.append((img, formulas[int(y[1])-1]))
    return combined


def token_to_int(data, dictionary, train_data=False):
    ret = []
    for unit in data:
        img, formula = unit
        formula = formula.split(' ')
        new_formula = [dictionary['\\<start>']]
        for token in formula:
            if token in dictionary:
                new_formula.append(dictionary[token])
            elif train_data:
                dictionary[token] = len(dictionary)
                new_formula.append(dictionary[token])
            else:
                new_formula.append(dictionary['\\<unknown>'])

        new_formula.append(dictionary['\\<end>'])
        new_formula += (config['label_length'] - len(new_formula)) * [dictionary['\\<pad>']]
        ret.append((img, new_formula))
    return ret


def batchify(data):
    batches = []
    for i in range(0, len(data) - config['batch_size'], config['batch_size']):
        images = [x[0] for x in data[i:i + config['batch_size']]]

        labels = [x[1] for x in data[i:i + config['batch_size']]]
        batches.append((images, labels))

    return batches