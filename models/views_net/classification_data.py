import os
import numpy as np
from pytorch3d.io import load_obj
import torch
# datastructures
from pytorch3d.structures import Meshes, Textures


class ClassificationData(object):

    def __init__(self, root_dir, data, device):
        # self.dir = os.path.join(root_dir, data)
        self.classes, self.class_to_idx = self.find_classes(root_dir)
        self.paths = self.make_dataset_by_class(root_dir, self.class_to_idx, data)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.device = device

    def __getitem__(self, index):
        path = self.paths[index][0]
        label = self.paths[index][1]

        # Load the obj and ignore the textures and materials.
        verts, faces_idx, _ = load_obj(path)
        faces = faces_idx.verts_idx

        # center = verts.mean(0)
        # verts = verts - center
        # scale = max(verts.abs().max(0)[0])
        # verts = verts / scale

        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = Textures(verts_rgb=verts_rgb)
        # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )

        return mesh, label

    def __len__(self):
        return self.size

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.endswith("obj") and (root.count(phase) == 1):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        meshes.append(item)
        return meshes

    def train_test_split(self, dataset, indices, test_size, permute=True):
        """
        Return two sets of indices to create a train and test data set

        Parameters
        ----------
        dataset : Samples from a Pytorch dataset (img,class)
        indices : Indices of the samples in the full dataset
        test_size : size of the test set in percentage (e.g. 30)
        permute : True to have randomly permuted sets

        Returns
        -------
        Two sets of indices (test set, train set)
        """
        indices = np.array(indices)
        _, classes = zip(*dataset)
        # get classes for all inputs
        classes = list(classes)
        # Get unique classes
        unique_classes = np.unique(classes)
        # Create k empty folds
        split = [[], []]
        # For each unique class divide the number of element as the split size (stratified split)
        for i in range(len(unique_classes)):
            # Get all indices of the class i
            class_indices = np.array(np.where(classes == unique_classes[i])[0])
            class_indices = indices[class_indices]
            # Get permuted indices of the same classes
            if permute:
                class_indices = class_indices[np.random.permutation(range(len(class_indices)))]
            # Where to cut the dataset
            cut_point = int((class_indices.size / 100) * test_size)
            # Get split
            arr = np.split(class_indices, [cut_point])
            # Save cut
            split[0].extend(arr[0])
            split[1].extend(arr[1])
        # Get permuted indices over all classes
        split[0] = np.array(split[0])
        split[1] = np.array(split[1])
        if permute:
            split[0] = split[0][np.random.permutation(range(len(split[0])))]
            split[1] = split[1][np.random.permutation(range(len(split[1])))]
        return split[0].tolist(), split[1].tolist()
