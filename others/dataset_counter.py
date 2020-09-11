# !/usr/bin/python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def count_modelnet_samples(path, num_classes):
    class_count = np.zeros((2, num_classes))
    class_names = []
    i = 0
    for class_name in os.scandir(path):
        if class_name.is_dir():
            class_names.append(class_name.name)
            for j, dataset_name in enumerate(["train", "test"]):
                for obj in os.listdir(os.path.join(class_name.path, dataset_name)):
                    if obj.endswith(".off"):
                        class_count[j, i] += 1
            i += 1
    sort_class_names = sorted(class_names)
    sort_class_count1 = []
    sort_class_count2 = []
    for c in sort_class_names:
        sort_class_count1.append(class_count[0][class_names.index(c)])
        sort_class_count2.append(class_count[1][class_names.index(c)])
    class_names = sort_class_names
    class_count = np.array([sort_class_count1, sort_class_count2])
    class_perc = np.zeros((3, num_classes))
    for i in range(num_classes):
        class_perc[0, i] = class_count[0, i] / np.sum(class_count[0, :])
        class_perc[1, i] = class_count[1, i] / np.sum(class_count[1, :])
        class_perc[2, i] = (class_count[0, i] + class_count[1, i]) / (
                np.sum(class_count[0, :]) + np.sum(class_count[1, :]))
    class_perc = class_perc * 100
    cl = class_count[0, :] + class_count[1, :]
    print(np.min(cl))
    plt.bar(class_names, class_count[0, :] + class_count[1, :])
    plt.ylabel('Number of objects')
    plt.yticks(np.arange(0, np.max(class_count[0, :] + class_count[1, :]), 100))
    plt.xticks(rotation='vertical')
    plt.title('ModelNet number of object per class')
    plt.show()

    plt.bar(class_names, class_perc[2, :])
    plt.ylabel('Percentage')
    plt.yticks(np.arange(0, np.max(class_perc[2, :]), 1))
    plt.xticks(rotation='vertical')
    plt.title('ModelNet classes percentage')
    plt.show()

    plt.bar(class_names, class_perc[0, :])
    plt.ylabel('Percentage')
    plt.yticks(np.arange(0, np.max(class_perc[0, :]), 1))
    plt.xticks(rotation='vertical')
    plt.title('ModelNet train classes percentage')
    plt.show()

    plt.bar(class_names, class_perc[1, :])
    plt.ylabel('Percentage')
    plt.xticks(rotation='vertical')
    plt.yticks(np.arange(0, np.max(class_perc[1, :]), 1))
    plt.title('ModelNet test classes percentage')
    plt.show()
    return class_perc


if __name__ == '__main__':
    path = sys.argv[1]
    num_classes = sys.argv[2]
    count_modelnet_samples(path, int(num_classes))
