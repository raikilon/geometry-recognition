import os

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def under_sampler_to(dataset, indices, nview, elements):
    """
    Under sample the given dataset so that each class has the given number of elements.
    If the class has less than the given number it takes all the elements of that class

    Parameters
    ----------
    dataset : Samples from a Pytorch dataset (img,class)
    indices : Indices of the samples in the full dataset
    Returns
    -------
    Indices of a under sampled dataset
    """
    indices = np.array(indices)
    _, classes = zip(*dataset)
    classes = list(classes)
    unique_classes = np.unique(classes)
    small_dataset = []
    for i in range(len(unique_classes)):
        class_indices = np.array(np.where(classes == unique_classes[i])[0])
        class_indices = indices[class_indices]
        class_indices = random_permute_indices(class_indices, nview)
        # keep only the right number of elements (elements*nviews because all the images are already concatenated)
        if len(class_indices) > elements * nview:
            small_dataset.extend(class_indices[:elements * nview])
        else:
            small_dataset.extend(class_indices)

    return small_dataset


def random_under_sampler(classes, indices, nview):
    """
    Under sample the given dataset so that each class has the same number of elements

    Parameters
    ----------
    dataset : Samples from a Pytorch dataset (img,class)
    indices : Indices of the samples in the full dataset
    Returns
    -------
    Indices of a under sampled dataset
    """
    indices = np.array(indices)
    # _, classes = zip(*dataset)
    # classes = list(classes)
    unique_classes = np.unique(classes)
    small_dataset = []
    min_size = np.Inf
    for i in range(len(unique_classes)):
        class_indices = np.array(np.where(classes == unique_classes[i])[0])
        min_size = min(min_size, class_indices.size)
    min_size = int(min_size)
    for i in range(len(unique_classes)):
        class_indices = np.array(np.where(classes == unique_classes[i])[0])
        class_indices = indices[class_indices]
        class_indices = random_permute_indices(class_indices, nview)
        small_dataset.extend(class_indices[:min_size])

    return small_dataset


def random_over_sampler(classes, indices, nview):
    """
    Over sample the given dataset so that each class has the same number of elements

    Parameters
    ----------
    dataset : Samples from a Pytorch dataset (img,class)
    indices : Indices of the samples in the full dataset
    Returns
    -------
    Indices of a over sampled dataset
    """
    indices = np.array(indices)
    # _, classes = zip(*dataset)
    # classes = list(classes)
    unique_classes = np.unique(classes)
    big_dataset = []
    max_size = 0
    for i in range(len(unique_classes)):
        class_indices = np.array(np.where(classes == unique_classes[i])[0])
        max_size = max(max_size, class_indices.size)
    max_size = int(max_size)

    for i in range(len(unique_classes)):
        class_indices = np.array(np.where(classes == unique_classes[i])[0])
        class_indices = indices[class_indices]
        class_indices = random_permute_indices(class_indices, nview)
        bigger_set = class_indices.tolist()
        # Get number of real sample (total number of objects)
        class_nsamp = int(len(bigger_set) / nview)
        while len(bigger_set) < max_size:
            samp_pos = np.random.randint(0, class_nsamp) * nview
            bigger_set.extend(class_indices[samp_pos:samp_pos + nview].tolist())
        big_dataset.extend(bigger_set)

    return big_dataset


def generate_loader(dataset, batch_size, workers):
    """
    Generate a data loader

    Parameters
    ----------
    dataset : Dataset create with subset
    batch_size : Batch size
    workers : Number of workser

    Returns
    -------
    Pytorch data laoder
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=True, drop_last=True)


def generate_dataset(data, path, trans=[]):
    """
    Generated a dataset with normalization for ImageNet with ImageFolder. The given folder need to follow the structure
    of ImageFolder (see Pytorch documentation)

    Parameters
    ----------
    data : Path of the dataset
    path : Name of the subset (test or train or val)

    Returns
    -------
    Return pytorch dataset
    """
    # Normalization parameters for ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    #normalize = transforms.Normalize(mean=[np.mean([0.485, 0.456, 0.406])],
    #                                 std=[np.mean([0.229, 0.224, 0.225])])
    data_dir = os.path.join(data, path)

    trans.extend([
        #transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        normalize,
    ])

    # Creates a dataset from a folder without need to create a custom class
    dataset = datasets.ImageFolder(data_dir, transforms.Compose(trans))

    # Order images because all the elements of the same classes need to be close
    # Already done by ImageFolder
    # dataset[0].imgs = sorted(dataset[0].imgs)
    # dataset[0].samples = dataset[0].imgs

    return dataset


def k_fold(dataset, nview, permute=False, k=10):
    """
    Return k stratified folds

    Parameters
    ----------
    dataset : Full dataset samples (input,target)
    nview : Number of view of RotationNet
    permute : Permute indices of the folds (random folds)
    k : Number of folds

    Returns
    -------
    List of k folds which contain the indices of each fold
    """
    _, classes = zip(*dataset)
    # get classes for all inputs
    classes = list(classes)
    # Get unique classes
    unique_classes = np.unique(classes)
    # Create k empty folds
    folds = [[] for i in range(k)]
    # For each unique class divide the number of element equally on all folds (stratified k-fold)
    for i in range(len(unique_classes)):
        # Get all indices of the class i
        indices = np.array(np.where(classes == unique_classes[i])[0])
        # Get permuted indices of the same class
        if permute:
            indices = random_permute_indices(indices, nview)
        # Stride of how many elements to jump for each fold
        stride = int((indices.size / nview) / k) * nview
        # Get point to where cut the indices to create k folds
        steps = np.arange(0, stride * k, stride)
        # Get folds
        arr = np.split(indices, steps[1:])
        for j in range(k):
            folds[j].extend(arr[j])
    # Get permuted indices over all classes
    if permute:
        for j in range(k):
            folds[j] = random_permute_indices(np.array(folds[j]), nview).tolist()
    return folds


def train_test_split(dataset, indices, nview, test_size, permute=True):
    """
    Return two sets of indices to create a train and test data set

    Parameters
    ----------
    dataset : Samples from a Pytorch dataset (img,class)
    indices : Indices of the samples in the full dataset
    nview : Number of views
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
            class_indices = random_permute_indices(class_indices, nview)
        # Where to cut the dataset
        cut_point = int((class_indices.size / nview) / (100 / test_size)) * nview
        # Get split
        arr = np.split(class_indices, [cut_point])
        # Save cut
        split[0].extend(arr[0])
        split[1].extend(arr[1])
    # Get permuted indices over all classes
    if permute:
        split[0] = random_permute_indices(np.array(split[0]), nview).tolist()
        split[1] = random_permute_indices(np.array(split[1]), nview).tolist()
    return split[0], split[1]


def random_permute_indices(indices, nview, internal_shuffle=False):
    """
    Permute the given indices

    Parameters
    ----------
    indices : numpy array of indices
    nview : number of views of RotationNet

    Returns
    -------
    Permuted indices
    """
    # Indices should be divisible by nview
    assert indices.size % nview == 0
    # Get number of real sample (total number of objects)
    train_nsamp = int(indices.size / nview)
    # Random permutation
    inds = np.zeros((nview, train_nsamp)).astype('int')
    # Get random permutation of the indices of the first view of each sample
    inds[0] = indices[np.random.permutation(range(train_nsamp)) * nview]
    # Put all the near views close to the first object
    if internal_shuffle:
        perm = np.random.permutation(range(1, nview))
        for batch_idx in range(0, nview - 1):
            inds[batch_idx + 1] = inds[0] + perm[batch_idx]
    else:
        for batch_idx in range(1, nview):
            inds[batch_idx] = inds[0] + batch_idx
    # Now all the near views are close to each other
    inds = inds.T.reshape(nview * train_nsamp)
    return inds
