#!/usr/bin/env python
import sys, os, argparse, logging
import numpy as np
import PIL
from PIL import Image

def process_args(args):
    parser = argparse.ArgumentParser(description='Process im2latex-100k train, test, development files (<label_idx> <img_path> <mode>) for formatting files such that can be used for training. (<img_path> <label_idx>>). Additionaly, large images, too long formulas and formulas that cannot be parsed will be discarded.')

    parser.add_argument('--image-dir', dest='image_dir',
                        type=str, required=True,
                        help=('Directory containing processed images.'
                        ))
    parser.add_argument('--data-path', dest='data_path',
                        type=str, required=True,
                        help=('Input file path containing <label_idx> <img_path> <mode> per line.'
                        ))
    parser.add_argument('--output-path', dest='output_path',
                        type=str, required=True,
                        help=('Output file path containing <img_path> <label_idx> per line.'
                        ))
    parser.add_argument('--label-path', dest='label_path',
                        type=str, required=True,
                        help=('Input label path containing <formula> per line.'
                        ))
    parser.add_argument('--max-width', dest='max_width',
                        type=int, default=625,
                        help=('Images with width than max-width will be discarded in the output file.'
                        ))
    parser.add_argument('--max-height', dest='max_height',
                        type=int, default=125,
                        help=('Images with larger height than max-width will be discarded in the output file.'
                        ))
    parser.add_argument('--max-tokens', dest='max_tokens',
                        type=int, default=200,
                        help=('Formulas with more than max-tokens tokens will be discarded in the output file.'
                        ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt' 
                        ))
    parser.add_argument('--postfix', dest='postfix',
                        type=str, default='.png',
                        help=('The format of images, default=".png".'
                        ))
    parameters = parser.parse_args(args)
    return parameters

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

    logging.info('Script being executed: %s'%__file__)
    data_path = parameters.data_path
    output_path = parameters.output_path
    image_dir = parameters.image_dir

    num_discard = 0
    num_nonexist = 0

    assert os.path.isfile(parameters.label_path), parameters.label_path
    labels = open(parameters.label_path).readlines()
    with open(output_path, 'w') as fout:
        with open(data_path, 'r') as fdata:
            for line in fdata:
                line_strip = line.strip()
                if len(line_strip) > 0:
                    line_idx, img_path, mod = line_strip.split()
                    img_path = os.path.join(image_dir, img_path) + parameters.postfix
                    if not os.path.exists(img_path):
                        logging.warning('%s does not exist!'%os.path.basename(img_path))
                        num_nonexist += 1
                        continue
                    im = Image.open(img_path)
                    im_size = im.size
                    w = im_size[0]
                    h = im_size[1]
                    if w <= parameters.max_width and h <= parameters.max_height:
                        label = labels[int(line_idx)]
                        if len(label.strip()) == 0:
                            logging.info('%s discarded due to cannot-be-parsed formula!'%os.path.basename(img_path))
                            num_discard += 1
                            continue
                        if len(label.strip().split()) > parameters.max_tokens:
                            logging.info('%s discarded due to too many tokens!'%os.path.basename(img_path))
                            continue
                        fout.write('%s %s\n'%(os.path.basename(img_path),line_idx))
                    else:
                        logging.info('%s discarded due to large image size!'%os.path.basename(img_path))
                        num_discard += 1
    logging.info('%d discarded. %d not found in %s.'%(num_discard, num_nonexist, image_dir))


if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
