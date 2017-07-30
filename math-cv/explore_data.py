import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import os


def tokenized_formula_lengths():
    f = open('dataset/normalized_formulas.lst').readlines()
    formula_lengths = [len(x) for x in f]
    return formula_lengths


def crop_image(img):
    boundary = np.where(img != 255)
    if np.count_nonzero(boundary) > 0:
        return np.max(boundary[0]) - np.min(boundary[0]), np.max(boundary[1]) - np.min(boundary[1])
    return None


def cropped_image_sizes():
    width = []
    height = []
    for image_file in os.listdir('dataset/formula_images/'):
        if not image_file.endswith(".png"):
            continue
        original_image = np.asarray(misc.imread("dataset/formula_images/" + image_file, True))
        cropped_image = crop_image(original_image)
        if cropped_image is not None:
            width.append(cropped_image[0])
            height.append(cropped_image[1])

    return width, height


def plot_histograms(datasets):
    for dataset in datasets:
        plt.figure(datasets.index(dataset))
        plt.hist(dataset[0])
        plt.title(dataset[1])
        plt.savefig('dataset/' + dataset[1] + '.png')


if __name__ == '__main__':
    histogram_datasets = []
    histogram_datasets.append((tokenized_formula_lengths(), 'formula_lengths'))
    images_sizes = cropped_image_sizes()
    histogram_datasets.append((images_sizes[0], 'image_heights'))
    histogram_datasets.append((images_sizes[1], 'image_widths'))
    plot_histograms(histogram_datasets)

