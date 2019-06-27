"""
This file is used to evaluate the performance of the model. Please run the test.py before running this file.

@ Author: Yu Sun. vxallset@outlook.com

@ Date created: Jun 04, 2019

@ Last modified: Jun 27, 2019

"""
import numpy as np


def calculate_sigma2s(distances):
    sigma2s = np.zeros(14, dtype=np.float)
    for keypoint_id in range(14):
        distance = distances[:, keypoint_id]
        distance2 = distance ** 2
        sigma2s[keypoint_id] = np.mean(distance2)
    return sigma2s


def calculate_OKS(distances, classes):
    sigma2s = calculate_sigma2s(distances)
    sigmas = np.sqrt(sigma2s)
    oks = np.zeros(len(distances))
    for id in range(len(distances)):
        one_distance = distances[id]
        one_class = classes[id]
        one_oks = np.sum(np.exp(-one_distance ** 2 / (2.0 * (1 * sigmas) ** 2)) *
                         np.array(one_class != 3, dtype=np.int)) / np.sum(np.array(one_class != 3, dtype=np.int))
        oks[id] = one_oks

    return oks


if __name__ == '__main__':
    distance_file = 'distances.npy'
    classes_file = 'classes.npy'
    distances = np.load(distance_file)
    classes = np.load(classes_file)
    oks = calculate_OKS(distances, classes)
    oks50_mask = np.array(oks > 0.5, dtype=np.int)
    oks75_mask = np.array(oks > 0.75, dtype=np.int)
    ap50 = np.sum(oks50_mask) / len(oks50_mask)
    ap75 = np.sum(oks75_mask) / len(oks75_mask)
    print("AP50 = {}, AP75 = {}".format(ap50, ap75))
