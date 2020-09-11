import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import time
from mvcnn import MVCNN
from svcnn import SVCNN
import uuid
import wandb
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

# add rotation net packages
sys.path.insert(0, os.path.abspath('../rotation_net'))
from utils.dataset import generate_dataset
from utils.dataset import random_over_sampler
from utils.dataset import train_test_split
from utils.dataset import random_under_sampler
from utils.dataset import random_permute_indices
from utils.model import load_model
from utils.model import save_model
from utils.statistics import Statistics
from utils.statistics import log_data
from utils.statistics import log_summary

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='PATH', help='path to dataset')
parser.add_argument('--train_type', default="hold-out", type=str, help="Type of training",
                    choices=["k-fold", "hold-out", "full", "evaluate"])
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--num_classes', default=40, type=int, help='number of classes (default: 40)')
parser.add_argument('--epochs', default=1000, type=int,
                    help='number of total epochs to run (early stopping is used to stop before this limit)')
parser.add_argument('--batch_size', default=200, type=int, help='mini-batch size (default: 200)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--model_path', type=str, help='path of the model to test')
parser.add_argument('--pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--case', default='2', type=str, help='viewpoint setup case 1, 2 or 3 (default: 2)')
parser.add_argument('--patience', default='20', type=int, help='patience of early stopping (default 50)')
parser.add_argument('--feature_extraction', action='store_true', help='Do feature extraction (train only classifier)')
parser.add_argument('--val_split_size', default=20, type=int, help='Size of the validation set (default: 20%)')
parser.add_argument('--optimizer', default='ADAM', help='model optimizer (default: ADAM)')
parser.add_argument('--sampling', default='no-sampling', choices=['oversampling', 'undersampling', 'nosampling'],
                    help='over sampling or under sampling (default: no-sampling)')
parser.add_argument('--learning_rate', default=0.00001, type=float, help='initial learning rate (default: 0.01)')
parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--arch', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')


def main():
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    if args.case == '1':
        args.nview = 12
    elif args.case == '2':
        args.nview = 20
    elif args.case == '3':
        args.nview = 160

        # Names for the saved checkpoints
    name = uuid.uuid1()
    args.fname_best = 'mvcnn{}_model_best{}.pth.tar'.format(args.nview, name)
    args.fname = 'mvcnn{}_model{}.pth.tar'.format(args.nview, name)

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    criterion = nn.CrossEntropyLoss().to(device=args.device)

    if args.train_type == 'hold-out':
        # Get full train dataset
        full_set = generate_dataset(args.data, 'train')


        full_set_indices = range(len(full_set[0].samples))

        # Get test train split. The test set here is intended as validation set but because we need also a validation set
        # for the early stopping we call it test
        test_idx, train_idx = train_test_split([full_set[0].samples[i] for i in full_set_indices], full_set_indices,
                                               args.nview,
                                               args.val_split_size, True)

        # Get val train split
        val_idx, train_idx = train_test_split([full_set[0].samples[i] for i in train_idx], train_idx, args.nview,
                                              args.val_split_size, True)

        # Get subsets
        test_set = torch.utils.data.Subset(full_set[0], test_idx)
        val_set = torch.utils.data.Subset(full_set[0], val_idx)
        train_set = torch.utils.data.Subset(full_set[0], train_idx)

        model, optimizer = create_model(args)
        # train_hold_out(model, criterion, optimizer, args, train_set, val_set, test_set, True)
        model, optimizer = create_model(args, model)
        train_hold_out(model, criterion, optimizer, args, train_set, val_set, test_set, False)

    os.remove(args.fname_best)


def create_model(args, model=None):
    # Create MVCCN model based on the given architecture.
    if model is None:
        model = SVCNN(nclasses=args.num_classes, pretraining=args.pretrained, cnn_name=args.arch,
                      feature_extraction=args.feature_extraction)
    else:
        model = MVCNN(model, num_views=args.nview)

    # Multi GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # Send model to GPU or keep it to the CPU
    model = model.to(device=args.device)

    if args.optimizer == "ADAM":
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate,
                         weight_decay=args.weight_decay)
    elif args.optimizer == "ADAGRAD":
        optimizer = Adagrad(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate,
                            weight_decay=args.weight_decay)
    else:
        # If we use feature extraction (features weights are frozen), we need to keep only differentiable params
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate,
                        momentum=args.momentum, weight_decay=args.weight_decay)
    return model, optimizer


def train_hold_out(model, criterion, optimizer, args, train_set, val_set, test_set, single_view=False):
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

    # Generate loaders
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=single_view, num_workers=args.workers,
                             pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=single_view, num_workers=args.workers,
                            pin_memory=True, drop_last=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=single_view, num_workers=args.workers,
                              pin_memory=True, drop_last=True)

    # Track model in wandb
    #wandb.init(project="MVCNN", config=args, reinit=True)

    # wandb.watch(model)

    train(model, criterion, optimizer, train_loader, val_loader, args, single_view)

    load_model(model, args.fname_best)

    val_statistics = validate(test_loader, model, criterion, args, single_view)

    log_summary(val_statistics, "val", test_loader.dataset.dataset.classes)


def train(model, criterion, optimizer, train_loader, val_loader, args, single_view):
    # Best prediction
    best_prec1 = 0

    epoch_no_improve = 0

    for epoch in range(args.epochs):
        if not single_view:
            # Give random permutation to the images
            indices = train_loader.dataset.indices
            inds = random_permute_indices(np.array(indices), args.nview)
            train_loader.dataset.indices = np.asarray(inds)
            del indices
            del inds

        statistics = Statistics()

        # switch to train mode
        model.train()
        start_time = time.time()
        for batch_idx, (input_val, target_val) in enumerate(train_loader):
            loss, (prec1, prec5), y_pred, y_true = execute_batch(model, criterion, input_val,
                                                                 target_val, args, single_view)

            statistics.update(loss.detach().cpu().numpy(), prec1, prec5, y_pred, y_true)
            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del loss
            torch.cuda.empty_cache()

        elapsed_time = time.time() - start_time

        # Evaluate on validation set
        val_statistics = validate(val_loader, model, criterion, args, single_view)

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

                return


def execute_batch(model, criterion, input_val, target_val, args, single_view):
    input_val = input_val.to(device=args.device, non_blocking=True)

    if not single_view:
        # number of samples
        nsamp = int(target_val.size(0) / args.nview)
        # get a target for each sample
        t = target_val.numpy()
        t = [t[i] for i in range(0, len(t), args.nview)]
        target_val = torch.from_numpy(np.array(t))
    else:
        nsamp = target_val.size(0)

    target_val = target_val.to(device=args.device, non_blocking=True)

    # compute output
    output = model(input_val)

    loss = criterion(output, target_val)

    maxk = 5
    # get the sorted top k for each sample
    _, pred = output.topk(maxk, dim=1, sorted=True)
    pred = pred.t()
    correct = pred.eq(target_val.contiguous().view(1, -1).expand_as(pred))

    prec = []
    for k in (1, 5):
        correct_k = correct[:k].view(-1).float().sum(0)
        res = correct_k.mul_(100.0 / nsamp)
        prec.append(res.detach().cpu().numpy())

    predictions = pred[0].detach().cpu().numpy()
    real_target = target_val.detach().cpu().numpy()
    # cleaning variable for out of memory problems
    del pred, input_val, target_val, correct, output
    torch.cuda.empty_cache()
    return loss, prec, predictions, real_target


def validate(test_loader, model, criterion, args, single_view):
    statistics = Statistics()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_val, target_val) in enumerate(test_loader):
            loss, (prec1, prec5), y_pred, y_true = execute_batch(model, criterion, input_val,
                                                                 target_val, args, single_view)

            statistics.update(loss.data.cpu().numpy(), prec1, prec5, y_pred, y_true)

    return statistics


if __name__ == '__main__':
    main()
