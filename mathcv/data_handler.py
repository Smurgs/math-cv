import json
import numpy as np
from PIL import Image

from mathcv.config import config


class DataLoader:
    def __init__(self):
        self._train = None
        self._val = None
        self._test = None
        self.load_data()
        self._dictionary = json.loads(open(config['mapper_path'], 'r').read())

    def get_train_batches(self):
        return self._train

    def get_validation_batches(self):
        return self._val

    def get_test_batches(self):
        return self._test

    def get_vocab_size(self):
        return len(self._dictionary)

    def load_data(self):
        formulas = open(config['formula_path']).read().split('\n')
        train = open(config['train_path']).read().split('\n')[:-1]
        val = open(config['val_path']).read().split('\n')[:-1]
        test = open(config['test_path']).read().split('\n')[:-1]

        train = self.combine_x_and_y(train[:10], formulas)
        val = self.combine_x_and_y(val[:10], formulas)
        test = self.combine_x_and_y(test[:10], formulas)

        self._train = self.batchify(train)
        self._val = self.batchify(val)
        self._test = self.batchify(test)

    @staticmethod
    def combine_x_and_y(lst, formulas):
        combined = []
        for y in lst:
            y = y.split(' ')
            img = np.array(Image.open(config['img_dir'] + y[0]).convert('L'))
            img = img[..., np.newaxis]
            formula = json.loads(formulas[int(y[1])])
            combined.append((img, formula))
        return combined

    @staticmethod
    def batchify(data):
        batches = []
        for i in range(0, len(data) - config['batch_size'], config['batch_size']):
            images = [x[0] for x in data[i:i + config['batch_size']]]
            labels = [x[1] for x in data[i:i + config['batch_size']]]
            batches.append((images, labels))
        return batches


if __name__ == '__main__':
    dl = DataLoader()
