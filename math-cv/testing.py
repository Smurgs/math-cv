import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import re

IM_2_LATEX_DATASET_PATH = "Dataset/"
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
    training_file = open(IM_2_LATEX_DATASET_PATH + "im2latex_train.lst").readlines()
    training_image_file_names = [x.split()[1] for x in training_file]
    for x in range(10000):
        img = np.asarray(misc.imread(IM_2_LATEX_DATASET_PATH + "formula_images/" + training_image_file_names[x] + ".png", True))
        crop_image(img)
        print x


def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def clean_formula(formula):
    # Uncomment line
    if formula[0] == "%":
        formula = formula[1:]
    formula = formula.split("%")[0]
    formula = formula.replace("$", "")
    formula = formula.replace("\\>", "")
    formula = formula.replace("\\;", "")
    formula = formula.replace("\\rm", "")
    formula = formula.replace(" ", "")                  # Spaces
    formula = formula.replace("\t", "")                  # Tabs
    formula = re.sub(r'\\label{.*?}', "", formula)
    formula = formula[:-1]
    formula = formula + "\\eos"
    return formula


def prepare_formulas():
    formulas = open(IM_2_LATEX_DATASET_PATH + "im2latex_formulas.lst").readlines()
    vocabulary = open(IM_2_LATEX_DATASET_PATH + "latex_vocab.lst").readlines()

    cleaned_formulas = [clean_formula(x) for x in formulas]

    mapper = [(x[:-1], vocabulary.index(x)) for x in vocabulary]

    integerized_formulas = []
    bad_strings = []
    for formula in cleaned_formulas:
        if cleaned_formulas.index(formula) % 1000 == 0:
            print cleaned_formulas.index(formula)
        str_formula = formula
        str_old = ""
        int_formula = ""
        while len(str_formula) > 0:
            if str_old == str_formula:
                bad_strings.append((str_formula, cleaned_formulas.index(formula)))
                break
            str_old = str_formula
            for vocab, mapping in mapper:
                if str_formula.startswith(vocab):
                    int_formula = int_formula + str(mapping) + " "
                    str_formula = str_formula[len(vocab):]
                    break
            # raise Exception("Failed to find mapping")
        int_formula = int_formula[:-1]
        integerized_formulas.append(int_formula)

    outfile = open("integerized_formulas.txt", "w")
    for x,y in bad_strings:
        outfile.write("---INDEX " + str(y) + "---:" + x)


if __name__ == "__main__":
    prepare_formulas()
