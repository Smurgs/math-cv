import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import re

DATASET = "dataset/"
TARGET = DATASET + "target/"

width = []
height = []


def prepare_image(img):
    img = crop_image(img)
    img = pad_image(img)
    pass


def pad_image(img):
    return 1


def crop_image(img):
    global width
    global height
    boundary = np.where(img != 255)
    if np.count_nonzero(boundary) > 0:
        cropped_img = img[np.min(boundary[0]):np.max(boundary[0]), np.min(boundary[1]):np.max(boundary[1])]
        width.append(np.max(boundary[0]) - np.min(boundary[0]))
        height.append(np.max(boundary[1]) - np.min(boundary[1]))
    # show_img(cropped_img)
        return cropped_img
    return img


def prepare_im2latex_dataset():
    training_file = open(DATASET + "im2latex_train.lst").readlines()
    training_image_file_names = [x.split()[1] for x in training_file]
    for x in range(10000):
        img = np.asarray(misc.imread(DATASET + "formula_images/" + training_image_file_names[x] + ".png", True))
        crop_image(img)
        print x


def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def clean_formula(formula):
    formula = formula[:-1]
    # Uncomment line
    if formula[0] == "%":
        formula = formula[1:]
    formula = formula.split("%")[0]
    formula = formula.replace("$", "")
    formula = formula.replace("\\>", "")
    formula = formula.replace("\\;", "")
    formula = formula.replace("\\rm", "")
    formula = formula.replace("\\ ", "")
    formula = formula.replace(" ", "")                  # Spaces
    formula = formula.replace("\t", "")                  # Tabs
    formula = re.sub(r'\\label{.*?}', "", formula)
    formula = formula + "\\eos"
    return formula


def prepare_formulas():
    formulas = open(DATASET + "im2latex_formulas.lst").readlines()
    vocabulary = open(DATASET + "latex_vocab.lst").readlines()

    cleaned_formulas = [clean_formula(x) for x in formulas[:100000]]
    mapper = [(x[:-1], vocabulary.index(x)) for x in vocabulary]

    integerized_formulas = []
    tokenized_formulas = []
    bad_strings = []
    for formula in cleaned_formulas:
        str_formula = formula
        str_old = ""
        int_formula = ""
        tokenized_formula = ""
        while len(str_formula) > 0:
            if str_old == str_formula:
                bad_strings.append(str_formula + "[" + str(cleaned_formulas.index(formula)) + "]")
                break
            str_old = str_formula
            for vocab, mapping in mapper:
                if str_formula.startswith(vocab):
                    int_formula = int_formula + str(mapping) + " "
                    tokenized_formula = tokenized_formula + vocab + " "
                    str_formula = str_formula[len(vocab):]
                    break
        int_formula = int_formula[:-1]
        integerized_formulas.append(int_formula)
        tokenized_formulas.append(tokenized_formula)

    outfile = open(TARGET + "integerized_formulas.txt", "w")
    outfile.write("\n".join(integerized_formulas))

    outfile = open(TARGET + "tokenized_formulas.txt", "w")
    outfile.write("\n".join(tokenized_formulas))

    outfile = open(TARGET + "bad_formulas.txt", "w")
    outfile.write("\n".join(bad_strings))


if __name__ == "__main__":
    prepare_formulas()
