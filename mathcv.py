import os
import subprocess
import tensorflow as tf

import mathcv.model
import mathcv.train
import mathcv.preprocess
import mathcv.data_handler
from mathcv.config import config


def main(argv):
    print ('MathCv v0.1.0')
    if len(argv) <= 1:
        print_usage()
    elif argv[1] == 'download':
        download_dataset()
    elif argv[1] == 'preprocess':
        mathcv.preprocess.preprocess_dataset()
    elif argv[1] == 'train':
        train_model()
    else:
        print_usage()


def print_usage():
    print ('Usage: mathcv command')
    print ('Commands:')
    print ('    download            download the im2latex dataset')
    print ('    preprocess          preprocess the train/val/test dataset')
    print ('    train               train the mathcv model')


def download_dataset():
    dataset_dir = os.path.join(config['root_dir'], 'dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    os.chdir(dataset_dir)
    print ('Downloading formulas')
    subprocess.call('curl -O https://zenodo.org/record/56198/files/im2latex_formulas.lst', shell=True)
    print ('Downloading dataset partitions')
    subprocess.call('curl -O https://zenodo.org/record/56198/files/im2latex_train.lst', shell=True)
    subprocess.call('curl -O https://zenodo.org/record/56198/files/im2latex_validate.lst', shell=True)
    subprocess.call('curl -O https://zenodo.org/record/56198/files/im2latex_test.lst', shell=True)
    print ('Downloading images')
    subprocess.call('curl -O https://zenodo.org/record/56198/files/formula_images.tar.gz', shell=True)
    print ('Extracting images from archive')
    subprocess.call('tar -xzvf formula_images.tar.gz', shell=True)
    os.remove('formula_images.tar.gz')


def train_model():
    mathcv.train.train()


def infer():
    pass


if __name__ == '__main__':
    tf.app.run()
