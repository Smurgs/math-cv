import json
import numpy as np
from PIL import Image

from mathcv.config import config


class Iterator:
    def __init__(self, iterable):
        self._iterable = iterable
        self._number_items = len(iterable)
        self._position = 0

    def get_next(self):
        if self._position == self._number_items:
            raise EOFError
        ret = self._iterable[self._position]
        self._position += 1
        return ret

    def reset(self):
        self._position = 0


class DataLoader:
    def __init__(self):
        self._train_iter = None
        self._val_iter = None
        self._test_iter = None
        self._formulas = None
        self._dictionary = json.loads(open(config['mapper_path'], 'r').read())

    def _get_formulas(self):
        if self._formulas is None:
            self._formulas = open(config['formula_path']).read().split('\n')
        return self._formulas

    def get_train_iterator(self):
        if self._train_iter is None:
            train = open(config['train_path']).read().split('\n')[:-1]
            if config['train_limit'] is not None:
                train = train[:config['train_limit']]

            train = self._combine_data(train)
            train = self._batchify(train)
            self._train_iter = Iterator(train)

        return self._train_iter

    def get_validation_iterator(self):
        if self._val_iter is None:
            val = open(config['val_path']).read().split('\n')[:-1]
            if config['val_limit'] is not None:
                val = val[:config['val_limit']]

            val = self._combine_data(val)
            val = self._batchify(val)
            self._val_iter = Iterator(val)

        return self._val_iter

    def get_test_iterator(self):
        if self._test_iter is None:
            test = open(config['test_path']).read().split('\n')[:-1]
            if config['test_limit'] is not None:
                test = test[:config['test_limit']]

            test = self._combine_data(test)
            test = self._batchify(test)
            self._test_iter = Iterator(test)

        return self._test_iter

    def get_vocab_size(self):
        return len(self._dictionary)

    def _combine_data(self, lst):
        combined = []
        for y in lst:
            y = y.split(' ')
            img = np.array(Image.open(config['img_dir'] + y[0]).convert('L'))
            img = img[..., np.newaxis]
            input = json.loads(self._get_formulas()[int(y[1])])
            target = input[1:] + [self._dictionary['<pad>']]
            combined.append((img, input, target))
        return combined

    @staticmethod
    def _batchify(data):
        batches = []
        for i in range(0, len(data) - config['batch_size'], config['batch_size']):
            images = [x[0] for x in data[i:i + config['batch_size']]]
            inputs = [x[1] for x in data[i:i + config['batch_size']]]
            targets = [x[2] for x in data[i:i + config['batch_size']]]
            batches.append((images, inputs, targets))
        return batches


if __name__ == '__main__':
    dl = DataLoader()
    i = dl.get_train_iterator()
    try:
        while True:
            a = i.get_next()
    except EOFError:
        print ('caught end')

