import logging
import time

import numpy as np
import torch
import torch.nn.functional
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import wandb

from test import validate
from utils.dataset import generate_dataset
from utils.dataset import generate_loader
from utils.dataset import k_fold
from utils.dataset import random_over_sampler
from utils.dataset import random_permute_indices
from utils.dataset import random_under_sampler
from utils.dataset import train_test_split
from utils.dataset import under_sampler_to
from utils.depth_dataset import DepthDataset
from utils.aligned_dataset import AlignedDataset
from utils.model import load_model
from utils.model import save_model
from utils.statistics import Statistics
from utils.statistics import log_data
from utils.statistics import log_summary
from utils.train import execute_batch
from utils.train import execute_batch_aligned

logger = logging.getLogger('RotationNet')


def train_hold_out(model, criterion, optimizer, args):
    """
    Perform a train with validation train split on the given model.  Use args.val_split_size to set the percentage of
    data for the validation set. Keep in mind the two validation sets are genereted. One for early stopping and one to
    evaluate the model.

    All the data are sent to wandb or logged on a text file

    Parameters
    ----------
    model : RotaitonNet model
    criterion : Pytorch criterion (CrossEntropy for RotationNet)
    optimizer : Pytorch optimizer (e.g. SGD)
    args : Input args from the parser

    Returns
    -------
    Nothing
    """

    trans = []

    if args.flip:
        # flip is with 0.5 probability
        trans.append(transforms.RandomHorizontalFlip())
        trans.append(transforms.RandomVerticalFlip())
    if args.rotation:
        trans.append(transforms.RandomRotation((0, 90), expand=args.rotation_expand))
    # if expand is true then resize the image to the right size
    if args.rotation_expand:
        trans.append(transforms.Resize(224))

    # Get full train dataset
    # Create two different sets but one with and one without transformations (random rotation and flip)
    if args.depth:
        # create one element tuple to be consistent with ImageFolder dataset of Pytorch
        full_set_train = (DepthDataset(args.data, 'train', trans),)
        full_set_val = (DepthDataset(args.data, 'train'),)
    else:
        full_set_train = generate_dataset(args.data, 'train', trans)
        full_set_val = generate_dataset(args.data, 'train',[])

    # Get test train split. The test set here is intended as validation set but because we need also a validation set
    # for the early stopping we call it test
    test_idx, train_idx = train_test_split(full_set_train[0].samples, range(len(full_set_train[0].samples)),
                                           args.nview,
                                           args.val_split_size, True)
    # Get val train split
    val_idx, train_idx = train_test_split([full_set_train[0].samples[i] for i in train_idx], train_idx, args.nview,
                                          args.val_split_size, True)

    if args.nelements != 0:
        train_idx = under_sampler_to([full_set_train[0].samples[i] for i in train_idx], train_idx, args.nview,
                                     args.nelements)

    _, classes = zip(*[full_set_train[0].samples[i] for i in train_idx])

    if args.sampling == "oversampling":
        train_idx = random_over_sampler(classes, train_idx, args.nview)
    elif args.sampling == "undersampling":
        train_idx = random_under_sampler(classes, train_idx, args.nview)

    # Get subsets
    test_set = torch.utils.data.Subset(full_set_val[0], test_idx)
    val_set = torch.utils.data.Subset(full_set_val[0], val_idx)
    train_set = torch.utils.data.Subset(full_set_train[0], train_idx)

    # Generate loaders
    test_loader = generate_loader(test_set, args.batch_size, args.workers)
    val_loader = generate_loader(val_set, args.batch_size, args.workers)
    train_loader = generate_loader(train_set, args.batch_size, args.workers)

    # load already trained model from single image classification
    # load_model(model, "for_rotation_net.pth.tar")

    # Track model in wandb
    wandb.init(project="RotationNet", config=args)

    # wandb.watch(model)

    train(model, criterion, optimizer, train_loader, val_loader, args)

    # Load best model before validating
    load_model(model, args.fname_best)

    # permute indices
    # indices = test_loader.dataset.indices
    # inds = random_permute_indices(np.array(indices), args.nview, True)
    # test_loader.dataset.indices = np.asarray(inds)
    # del indices
    # del inds

    val_statistics = validate(test_loader, model, criterion, args)

    log_summary(val_statistics, "val", test_loader.dataset.dataset.classes)


def train_hold_out_aligned(model, criterion, optimizer, args):
    """
    Perform a train with validation train split on the given model.  Use args.val_split_size to set the percentage of
    data for the validation set. Keep in mind the two validation sets are genereted. One for early stopping and one to
    evaluate the model.

    All the data are sent to wandb or logged on a text file

    Parameters
    ----------
    model : RotaitonNet model
    criterion : Pytorch criterion (CrossEntropy for RotationNet)
    optimizer : Pytorch optimizer (e.g. SGD)
    args : Input args from the parser

    Returns
    -------
    Nothing
    """

    # create one element tuple to be consistent with ImageFolder dataset of Pytorch
    full_set_train = (AlignedDataset(args.data_aligned, 'train', []),)
    # validation data is on normal dataset

    full_set_val = generate_dataset(args.data, 'train')
    # full_set_val = (AlignedDataset(args.data_aligned, 'train', []),)
    # Get test train split. The test set here is intended as validation set but because we need also a validation set
    # for the early stopping we call it test
    test_idx, train_idx = train_test_split(full_set_train[0].samples, range(len(full_set_train[0].samples)),
                                           args.nview,
                                           args.val_split_size, True)
    # Get val train split
    val_idx, train_idx = train_test_split([full_set_train[0].samples[i] for i in train_idx], train_idx, args.nview,
                                          args.val_split_size, True)

    _, classes = zip(*[full_set_train[0].samples[i] for i in train_idx])

    # Get subsets
    test_set = torch.utils.data.Subset(full_set_val[0], test_idx)
    val_set = torch.utils.data.Subset(full_set_val[0], val_idx)
    train_set = torch.utils.data.Subset(full_set_train[0], train_idx)

    # Generate loaders
    test_loader = generate_loader(test_set, args.batch_size, args.workers)
    val_loader = generate_loader(val_set, args.batch_size, args.workers)
    train_loader = generate_loader(train_set, args.batch_size, args.workers)

    # load already trained model from single image classification
    # load_model(model, "for_rotation_net.pth.tar")

    # Track model in wandb
    wandb.init(project="RotationNet", config=args)

    # wandb.watch(model)

    train(model, criterion, optimizer, train_loader, val_loader, args)

    # Load best model before validating
    load_model(model, args.fname_best)

    # permute indices
    # indices = test_loader.dataset.indices
    # inds = random_permute_indices(np.array(indices), args.nview, True)
    # test_loader.dataset.indices = np.asarray(inds)
    # del indices
    # del inds

    val_statistics = validate(test_loader, model, criterion, args)

    log_summary(val_statistics, "val", test_loader.dataset.dataset.classes)


def train_k_fold(model, criterion, optimizer, args):
    """
    Perform a k-fold cross validation on the given model.  Use args.fold to set the number of total folds and
    args.ntest_fold to set the number of folds dedicated for the validation. Keep in mind that each training step uses
    also an extra validation set for early stopping (the size of this subset is set via args.val_split_size).

    All the data are sent to wandb or logged on a text file

    Parameters
    ----------
    model : RotaitonNet model
    criterion : Pytorch criterion (CrossEntropy for RotationNet)
    optimizer : Pytorch optimizer (e.g. SGD)
    args : Input args from the parser

    Returns
    -------
    Nothing
    """
    # Save clean model to reload at each fold
    save_model(args.arch, model, optimizer, args.fname)

    # Get full train dataset
    full_set = generate_dataset(args.data, 'train')

    # Get folds
    folds = k_fold(full_set[0].samples, args.nview, True, args.fold)

    # List of top1 and top5 accuracies to get the average and std of the model performance
    top1 = []
    top5 = []

    # K-fold cross validation
    for i in range(args.fold):
        test_idx = []
        # Use ntest_folds folds for the test set
        for j in range(args.ntest_folds):
            test_idx.extend(folds[i])
            folds = np.delete(folds, i, 0)

        # Use rest of the data for the train set
        train_idx = np.hstack(folds)
        val_idx, train_idx = train_test_split([full_set[0].samples[i] for i in train_idx], train_idx, args.nview,
                                              args.val_split_size, True)

        # Get subsets
        test_set = torch.utils.data.Subset(full_set[0], test_idx)
        val_set = torch.utils.data.Subset(full_set[0], val_idx)
        train_set = torch.utils.data.Subset(full_set[0], train_idx)

        # Generate loaders
        test_loader = generate_loader(test_set, args.batch_size, args.workers)
        val_loader = generate_loader(val_set, args.batch_size, args.workers)
        train_loader = generate_loader(train_set, args.batch_size, args.workers)

        logger.debug("Start train on fold {}/{}".format(i, args.fold))

        # Track model in wandb
        wandb.init(project="RotationNet", name="Fold " + str(i), config=args, reinit=True)

        # The model can be analyzed only once
        # if i == 0:
        #    wandb.watch(model)

        train(model, criterion, optimizer, train_loader, val_loader, args)

        # Load best model before validating
        load_model(model, args.fname_best)

        val_statistics = validate(test_loader, model, criterion, args)
        log_summary(val_statistics, "val", test_loader.dataset.dataset.classes)

        # Load fresh model for next train
        load_model(model, args.fname)

    logger.info('Val prec@1 {top1:.3f} +- {std1:.3f} \t'
                'Val prec@5 {top5:.3f} +- {std5:.3f} \t'.format(top1=np.mean(top1), std1=np.std(top1),
                                                                top5=np.mean(top5), std5=np.std(top5)))


def train_all(model, criterion, optimizer, args):
    """
    Train the model on the full dataset with the expect for the small validation set (size set via args.val_spit_size)
    for early stopping.

    All the data are sent to wandb or logged on a text file

    Parameters
    ----------
    model : RotaitonNet model
    criterion : Pytorch criterion (CrossEntropy for RotationNet)
    optimizer : Pytorch optimizer (e.g. SGD)x
    args : Input args from the parser

    Returns
    -------
    Nothing
    """

    # Get full train dataset
    full_set = generate_dataset(args.data, 'train')

    # Train the network on all train set but still need a val set for early stopping
    val_idx, train_idx = train_test_split(full_set[0].samples, range(len(full_set[0].samples)), args.nview,
                                          args.val_split_size)

    # Get subsets
    val_set = torch.utils.data.Subset(full_set[0], val_idx)
    train_set = torch.utils.data.Subset(full_set[0], train_idx)

    # Get loaders
    val_loader = generate_loader(val_set, args.batch_size, args.workers)
    train_loader = generate_loader(train_set, args.batch_size, args.workers)

    # Track model in wandb
    wandb.init(project="RotationNet", name="Full", config=args)
    # wandb.watch(model)

    train(model, criterion, optimizer, train_loader, val_loader, args)


def train(model, criterion, optimizer, train_loader, val_loader, args):
    """
    Train the model on the train data loader data and stop when the top1 precision did not increased for args.patience
    epochs. The early stopping is done on the validation data loader.

    All the data are sent to wandb or logged on a text file. More precisely for each epoch the validation and training
    performance are sent to wandb and every args.print_freq the batch performance are logged on a file. At the end of
    the training the best top1 validation accuracy is sent to wandb.

    Parameters
    ----------
    model : RotaitonNet model
    criterion : Pytorch criterion (CrossEntropy for RotationNet)
    optimizer : Pytorch optimizer (e.g. SGD)
    train_loader : Data loader with training data (this must be created with a subset)
    val_loader : Data loader with validation data for early stopping (this must be created with a subset)
    args : Input args from the parser

    Returns
    -------
    Nothing
    """
    # Best prediction
    best_prec1 = 0
    # Using lr_scheduler for learning rate decay
    # scheduler = StepLR(optimizer, step_size=args.learning_rate_decay, gamma=0.1)

    epoch_no_improve = 0

    for epoch in range(args.epochs):
        # Give random permutation to the images
        indices = train_loader.dataset.indices
        inds = random_permute_indices(np.array(indices), args.nview, False)
        train_loader.dataset.indices = np.asarray(inds)
        del indices
        del inds

        statistics = Statistics()

        # switch to train mode
        model.train()
        start_time = time.time()
        for batch_idx, (input_val, target_val) in enumerate(train_loader):
            # loss, (prec1, prec5), y_pred, y_true = execute_batch(model, criterion, input_val,
            #                                                     target_val, args)
            loss, (prec1, prec5), y_pred, y_true = execute_batch_aligned(model, criterion, input_val,
                                                                         target_val, args)

            statistics.update(loss.detach().cpu().numpy(), prec1, prec5, y_pred, y_true)
            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                logger.debug('Batch: [{0}/{1}]\t'
                             'Loss {loss:.4f} \t'
                             'Prec@1 {top1:.3f} \t'
                             'Prec@5 {top5:.3f}'.format(batch_idx, len(train_loader), loss=loss.data, top1=prec1,
                                                        top5=prec5))
            del loss
            torch.cuda.empty_cache()

        elapsed_time = time.time() - start_time

        logger.debug("Evaluating epoch {}".format(epoch))

        # permute indices
        # indices = val_loader.dataset.indices
        # inds = random_permute_indices(np.array(indices), args.nview)
        # val_loader.dataset.indices = np.asarray(inds)
        # del indices
        # del inds
        # Evaluate on validation set
        val_statistics = validate(val_loader, model, criterion, args)

        # statistics.compute(args.num_classes)
        # val_statistics.compute(args.num_classes)

        log_data(statistics, "train", val_loader.dataset.dataset.classes, epoch)
        log_data(val_statistics, "internal_val", val_loader.dataset.dataset.classes, epoch)

        wandb.log({"Epoch elapsed time": elapsed_time}, step=epoch)

        #  Save best model and best prediction
        if val_statistics.top1.avg > best_prec1:
            best_prec1 = val_statistics.top1.avg
            save_model(args.arch, model, optimizer, args.fname_best)
            epoch_no_improve = 0
        else:
            # Early stopping
            epoch_no_improve += 1
            if epoch_no_improve == args.patience:
                wandb.run.summary["best_internal_val_top1_accuracy"] = best_prec1
                wandb.run.summary["best_internal_val_top1_accuracy_epoch"] = epoch - args.patience

                logger.debug("Stopping at epoch {} for early stopping (best was at epoch {})".format(epoch,
                                                                                                     epoch - args.patience))
                return

        # learning rate decay
        # scheduler.step()
