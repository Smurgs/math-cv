import os
import shutil
import subprocess

from config import config


def preprocess_dataset():

    preprocess_formulas()

    # Preprocess images

    # Filter out unwanted data points


def preprocess_formulas():
    input_file = config['original_formula_path']
    output_file = config['formula_path']

    assert os.path.exists(input_file)
    if not os.path.exists(os.path.dirname(os.path.dirname(config['formula_path']))):
        os.makedirs(os.path.dirname(os.path.dirname(config['formula_path'])))
    if not os.path.exists(os.path.dirname(config['formula_path'])):
        os.makedirs(os.path.dirname(config['formula_path']))

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


def is_ascii(str):
    try:
        str.decode('ascii')
        return True
    except UnicodeError:
        return False


if __name__ == '__main__':
    preprocess_dataset()