import os
import PIL
import json
import glob
import shutil
import subprocess
import numpy as np
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool

from config import config


def preprocess_dataset():
    if not os.path.exists(os.path.dirname(os.path.dirname(config['formula_path']))):
        os.makedirs(os.path.dirname(os.path.dirname(config['formula_path'])))
    if not os.path.exists(os.path.dirname(config['formula_path'])):
        os.makedirs(os.path.dirname(config['formula_path']))

    preprocess_formulas()

    preprocess_images()

    filter_records()


def filter_records():
    train_input = config['original_train_path']
    val_input = config['original_val_path']
    test_input = config['original_test_path']
    train_output = config['train_path']
    val_output = config['val_path']
    test_output = config['test_path']
    img_dir = config['img_dir']

    for input_path, output_path in [(train_input, train_output), (val_input, val_output), (test_input, test_output)]:

        num_discard = 0
        num_nonexist = 0
        num_in_group = 0

        assert os.path.isfile(config['formula_path'])
        labels = open(config['formula_path']).readlines()
        with open(output_path, 'w') as fout:
            with open(input_path, 'r') as fdata:
                for line in fdata:
                    line_strip = line.strip()
                    if len(line_strip) > 0:
                        line_idx, img_path, mod = line_strip.split()
                        img_path = os.path.join(img_dir, img_path) + config['image_postfix']
                        if not os.path.exists(img_path):
                            num_nonexist += 1
                            continue
                        im = Image.open(img_path)
                        im_size = im.size
                        w = im_size[0]
                        h = im_size[1]
                        if w <= config['image_width'] and h <= config['image_height']:
                            label = labels[int(line_idx) - 1]
                            if len(label.strip()) == 0:
                                num_discard += 1
                                continue
                            fout.write('%s %s\n' % (os.path.basename(img_path), line_idx))
                            num_in_group += 1
                        else:
                            num_discard += 1
        print('%d discarded. %d not found. Left with %d in %s.' % (num_discard, num_nonexist, num_in_group, output_path))





def preprocess_formulas():
    input_file = config['original_formula_path']
    output_file = config['formula_path']
    assert os.path.exists(input_file)

    cmd = "perl -pe 's|hskip(.*?)(cm\\|in\\|pt\\|mm\\|em)|hspace{\\1\\2}|g' %s > %s" % (input_file, output_file)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('FAILED: %s' % cmd)
        exit(1)

    temp_file = output_file + '.tmp'
    with open(temp_file, 'w+') as fout:
        with open(output_file) as fin:
            for line in fin:
                fout.write(line.replace('\r', ' ').strip() + '\n')  # delete \r

    js_script_path = os.path.join(config['root_dir'], 'mathcv/preprocess_latex.js')
    cmd = "cat %s | node %s %s > %s " % (temp_file, js_script_path, 'normalize', output_file)
    ret = subprocess.call(cmd, shell=True)
    os.remove(temp_file)
    if ret != 0:
        print('FAILED: %s' % cmd)
        exit(1)

    temp_file = output_file + '.tmp'
    shutil.move(output_file, temp_file)
    with open(temp_file) as fin:
        with open(output_file, 'w') as fout:
            for line in fin:
                tokens = line.strip().split()
                tokens_out = []
                for token in tokens:
                    if is_ascii(token):
                        tokens_out.append(token)
                fout.write(' '.join(tokens_out) + '\n')
    os.remove(temp_file)

    dictionary = dict()
    dictionary['<start>'] = len(dictionary)
    dictionary['<end>'] = len(dictionary)
    dictionary['<pad>'] = len(dictionary)
    shutil.move(output_file, temp_file)
    with open(temp_file) as fin:
        with open(output_file, 'w') as fout:
            for line in fin:
                tokens = line.strip().split()
                if len(tokens) > 200:
                    fout.write('\n')
                    continue
                tokens_out = [dictionary['<start>']]
                for token in tokens:
                    if token not in dictionary:
                        dictionary[token] = len(dictionary)
                    tokens_out.append(dictionary[token])
                tokens_out.append(dictionary['<end>'])
                tokens_out += (config['label_length'] - len(tokens_out)) * [dictionary['<pad>']]
                fout.write((str(tokens_out)) + '\n')
    os.remove(temp_file)

    json.dump(dictionary, open(config['mapper_path'], 'w'))


def preprocess_images():

    input_dir = config['original_imd_dir']
    output_dir = config['img_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    postfix = config['image_postfix']
    pad_size = json.loads(config['padding_size'])
    downsample_ratio = config['downsample_ratio']
    buckets = json.loads('[[' + str(config['image_width']*int(downsample_ratio)) + ',' + str(config['image_height']*int(downsample_ratio)) + ']]')

    filenames = glob.glob(os.path.join(input_dir, '*' + postfix))
    pool = ThreadPool(config['num_thread_preprocess'])
    results = pool.map(main_parallel, [(filename, postfix, os.path.join(output_dir, os.path.basename(filename)),
                                        pad_size, buckets, downsample_ratio) for filename in
                                       filenames])
    pool.close()
    pool.join()


def main_parallel(l):
    filename, postfix, output_filename, pad_size, buckets, downsample_ratio = l
    postfix_length = len(postfix)
    status = edit_image(filename, output_filename, pad_size, buckets, downsample_ratio)


def is_ascii(str):
    try:
        str.decode('ascii')
        return True
    except UnicodeError:
        return False


def edit_image(img, output_path, pad_size, buckets, ratio):
    # Crop image
    old_im = Image.open(img).convert('L')
    img_data = np.asarray(old_im, dtype=np.uint8)   # height, width
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:                       # If image is empty, don't alter
        old_im.save(output_path)
        return
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))      # Crop to fit formula

    # Pad and bucket image
    PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = pad_size
    old_size = (old_im.size[0]+PAD_LEFT+PAD_RIGHT, old_im.size[1]+PAD_TOP+PAD_BOTTOM)
    j = -1
    for i in range(len(buckets)):
        if old_size[0] <= buckets[i][0] and old_size[1] <= buckets[i][1]:
            j = i
            break
    if j < 0:
        new_size = old_size
        new_im = Image.new("RGB", new_size, (255, 255, 255))
        new_im.paste(old_im, (PAD_LEFT, PAD_TOP))
        new_im.save(output_path)
        return
    new_size = buckets[j]
    new_im = Image.new("RGB", new_size, (255, 255, 255))
    new_im.paste(old_im, (PAD_LEFT, PAD_TOP))

    # Downsample image
    assert ratio >= 1, ratio
    if ratio == 1:
        return
    old_im = new_im
    old_size = old_im.size
    new_size = (int(old_size[0]/ratio), int(old_size[1]/ratio))

    new_im = old_im.resize(new_size, PIL.Image.LANCZOS)
    new_im.save(output_path)
    return True


if __name__ == '__main__':
    preprocess_dataset()