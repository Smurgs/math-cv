import numpy as np
from scipy import misc
import linecache


def load_training_data():

    training_file = open("Dataset/im2latex_train.lst").readlines()
    training_label_list = [x.split()[0] for x in training_file]
    training_image_list = [x.split()[1] for x in training_file]

    training_data = {}
    training_data["x_train"] = []
    training_data["y_train"] = []
    for x in range(len(training_image_list[0:4])):
        training_data["x_train"].append(misc.imread("Dataset/formula_images/" + training_image_list[x] + ".png", True))
        training_data["y_train"].append(linecache.getline("Dataset/im2latex_formulas.lst", int(training_label_list[x])))

    training_data["x_train"] = np.asarray(training_data["x_train"])

    print training_data["x_train"].shape
    print training_data["y_train"]
    return training_data
