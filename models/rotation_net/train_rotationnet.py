import argparse
import distutils
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from rotation_net import RotationNet
from test import threshold_evaluation
from test import test_model
from train import train_all
from train import train_hold_out
from train import train_hold_out_aligned
from train import train_k_fold

# Log everything on a file
logger = logging.getLogger('rotation_net')
fh = logging.FileHandler("rotation_net{}.log".format(datetime.now().strftime("%d_%b_%Y_%H_%M_%S")))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


def str2bool(v):
    return bool(distutils.util.strtobool(v))


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Rotation Training')
parser.add_argument('data', metavar='PATH', help='path to dataset')
parser.add_argument('--data_aligned', type=str, help='path to aligned dataset')
parser.add_argument('--train_type', default="hold-out", type=str, help="Type of training",
                    choices=["k-fold", "hold-out", "full", "evaluate", "test","aligned"])
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--num_classes', default=40, type=int, help='number of classes (default: 40)')
parser.add_argument('--epochs', default=1000, type=int,
                    help='number of total epochs to run (early stopping is used to stop before this limit)')
parser.add_argument('--batch_size', default=200, type=int, help='mini-batch size (default: 400)')

parser.add_argument('--learning_rate_decay', default=100, type=float,
                    help='Number of epochs to decay learning rate (default: 100)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--model_path', type=str, help='path of the model to test')
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=False, help='use pre-trained model')
parser.add_argument('--flip', type=str2bool, nargs='?', const=True, default=False,
                    help='vertical and horizontal flip in train')
parser.add_argument('--rotation', type=str2bool, nargs='?', const=True, default=False, help='0 to 90 rotation in train')
parser.add_argument('--rotation_expand', type=str2bool, nargs='?', const=True, default=False,
                    help='expand image in rotation')
parser.add_argument('--case', default='2', type=str, help='viewpoint setup case 1, 2 or 3 (default: 2)')
parser.add_argument('--patience', default='20', type=int, help='patience of early stopping (default 50)')
parser.add_argument('--fold', default='10', type=int, help='number of fold for the cross validation (default 10)')
parser.add_argument('--debug', action='store_true', help='see debug messages')
parser.add_argument('--feature_extraction', type=str2bool, nargs='?', const=True, default=False,
                    help='Do feature extraction (train only classifier)')
parser.add_argument('--depth', type=str2bool, nargs='?', const=True, default=False,
                    help='Use depth depth images with rgb (default False)')
parser.add_argument('--print_freq', default=20, type=int, help='print frequency works only with debug (default: 20)')
parser.add_argument('--ntest_folds', default=2, type=int,
                    help='Number of fold for the validation set in k-fold (default: 2)')
parser.add_argument('--val_split_size', default=20, type=int, help='Size of the validation set (default: 20%)')
parser.add_argument('--target', type=int, help='Target for real time testing')
parser.add_argument('--threshold', type=int, help='inlier threshold')
parser.add_argument('--optimizer', default='SGD', help='model optimizer (default: SGD)')
parser.add_argument('--sampling', default='no-sampling', choices=['oversampling', 'undersampling', 'nosampling'],
                    help='over sampling or under sampling (default: no-sampling)')
parser.add_argument('--pooling', default='max', choices=['max', 'avg'], help='pooling layer (default: max)')
parser.add_argument('--nelements', default=0, type=int,
                    help='under sample to given number of elements. If a class contains fewer elements it takes all the elements of the class (default: 0 -> all elements)')
parser.add_argument('--learning_rate', default=0.01, type=float, help='initial learning rate (default: 0.01)')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--arch', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')


def main():
    args = parser.parse_args()
    # REPRODUCIBILITY
    torch.manual_seed(0)
    np.random.seed(0)

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Retrieve views candidates and right number of views
    if args.case == '1':
        args.vcand = np.load('view_candidates/vcand_case1.npy')
        args.nview = 12
    elif args.case == '2':
        args.vcand = np.load('view_candidates/vcand_case2.npy')
        args.nview = 20
    elif args.case == '3':
        args.vcand = np.load('view_candidates/vcand_case3.npy')
        args.nview = 160

    # Names for the saved checkpoints
    args.fname_best = 'rotationnet{}_model_best{}.pth.tar'.format(args.nview,
                                                                  datetime.now().strftime("%d_%b_%Y_%H_%M_%S"))
    args.fname = 'rotationnet{}_model{}.pth.tar'.format(args.nview, datetime.now().strftime("%d_%b_%Y_%H_%M_%S"))

    logger.debug("Number of view candidates: {}".format(np.shape(args.vcand)[0]))
    logger.debug("Number of views: {}".format(args.nview))

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    logger.debug("PyTorch is using  {}".format(args.device))

    # Mini batch size is used to do an update of the gradient so it need to be divisible by the number of views
    # otherwise one or more classification are not complete
    if args.batch_size % args.nview != 0:
        logger.error('Batch size should be multiplication of the number of views, {}'.format(args.nview))
        exit(1)

    # Get number of classes
    logger.debug("Number of classes: {}".format(args.num_classes))

    # Create RotationNet model based on the given architecture.
    # The output size is (num_classes + wrong_view class) * the number of views
    model = RotationNet(args.arch, args.pretrained, (args.num_classes + 1) * args.nview, args.feature_extraction,
                        args.depth)

    # Multi GPUs
    if torch.cuda.device_count() > 1:
        logger.debug("Using multiple GPUs")
        model = torch.nn.DataParallel(model)
    # Send model to GPU or keep it to the CPU
    model = model.to(device=args.device)

    # Define loss function (criterion) and optimizer
    # Sending loss to cuda is unnecessary because loss function is not stateful
    # TODO test if it works without sending loss to GPU
    criterion = nn.CrossEntropyLoss().to(device=args.device)

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

    # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
    # some boost when the network do not change
    # useless because cluster do not have cudnn
    # cudnn.benchmark = True

    logger.info("Model args: {}".format(args))

    if args.train_type == 'k-fold':
        logger.debug("K-fold training")
        train_k_fold(model, criterion, optimizer, args)
    elif args.train_type == 'hold-out':
        logger.debug("Hold-out training")
        train_hold_out(model, criterion, optimizer, args)
    elif args.train_type == 'full':
        logger.debug("Full training")
        train_all(model, criterion, optimizer, args)
    elif args.train_type == 'evaluate':
        logger.debug("Start evaluation on test set")
        test_model(model, criterion, args)
    elif args.train_type == 'aligned':
        logger.debug("Holt-out training on aligned set")
        train_hold_out_aligned(model, criterion,optimizer, args)
    elif args.train_type == "test":
        logger.debug("Start real time test")
        threshold_evaluation(model, args)

    # os.remove(args.fname_best)


if __name__ == '__main__':
    main()
