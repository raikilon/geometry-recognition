import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh
import numpy as np


class ClassificationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)
        self.classes, self.class_to_idx = self.find_classes(self.dir)
        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, opt.phase)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        label = self.paths[index][1]
        mesh = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder)
        meta = {'mesh': mesh, 'label': label}
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

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
                    if is_mesh_file(fname) and (root.count(phase) == 1):
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

    def reduce_dataset(self, idx, elements):
        # keep only n elements for the training
        indices = np.array(idx)
        _, classes = zip(*[self.paths[i] for i in idx])
        classes = np.array(list(classes))
        unique_classes = list(self.class_to_idx.values())
        small_dataset = []
        for i in range(len(unique_classes)):
            class_indices = np.array(np.where(classes == unique_classes[i])[0])
            class_indices = indices[class_indices]
            if len(class_indices) > elements:
                small_dataset.extend(class_indices[:elements])
            else:
                small_dataset.extend(class_indices)

        return small_dataset