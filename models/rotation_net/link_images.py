#!/usr/bin/python

import os
import sys


def generate_image_folder_dataset():
    """
    Transform ModelNet such that it is compatible with ImageFolder from Pytorch. The new dataset is created via links
    therefore, do not cancel the older folder.

    The input are given via argv

    Returns
    -------
    Nothing
    """
    dataset = sys.argv[1]
    output = sys.argv[2]

    os.makedirs("{}/train".format(output))
    os.makedirs("{}/test".format(output))

    for class_name in os.scandir(dataset):
        if class_name.is_dir():
            os.makedirs(os.path.join(output, "train", class_name.name))
            os.makedirs(os.path.join(output, "test", class_name.name))
            for dataset_name in ["train", "test"]:
                i = 0
                for img in sorted(os.listdir(os.path.join(class_name.path, dataset_name))):
                    # If there are other files than images it skip them
                    if img.endswith(".jpg") or img.endswith(".png"):
                        if "black_orthographic" in img or "depth_orthographic" in img:
                            i += 1
                            if len(sys.argv) > 3:
                                if (i - 1) % 4 != 0:
                                    continue

                            os.symlink(os.path.join("..", "..", "..", class_name.path, dataset_name, img),
                                       os.path.join(output, dataset_name, class_name.name, img))


if __name__ == '__main__':
    generate_image_folder_dataset()
