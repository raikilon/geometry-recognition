import numpy as np
import torch
import torch.nn.functional
import torch.nn.parallel
import torch.optim
import torch.utils.data
import wandb

from utils.dataset import generate_dataset
from utils.dataset import generate_loader
from utils.model import load_model
from utils.statistics import Statistics
from utils.statistics import log_summary
from utils.train import execute_batch
from utils.train import execute_inference


def threshold_evaluation(model, args):
    model = torch.nn.DataParallel(model)
    load_model(model, args.model_path, 'cpu')
    test_set = (generate_dataset(args.data, 'test'),)
    test_set = torch.utils.data.Subset(test_set[0], range(len(test_set[0].samples)))
    test_loader = generate_loader(test_set, args.batch_size, args.workers)

    # switch to evaluate mode
    model.eval()
    statistics = Statistics()
    with torch.no_grad():
        for batch_idx, (input_val, target_val) in enumerate(test_loader):
            predictions, score, target, top1 = execute_inference(model, input_val, target_val, args)
            statistics.update(0, top1, 0, (np.array(score) < args.threshold).astype(int), target)

    statistics.compute(classes=["inliers", "outliers"])
    # plt = generate_confusion_matrix(["inliers", "outliers"], statistics.confusion_matrix)
    # plt.savefig('confusion_matrix{}.png'.format(args.threshold))
    # plt.close()
    print("Threshold: {}".format(args.threshold))
    print("Accuracy: {}".format(statistics.top1.avg))
    print("Precision: {}".format(statistics.weighted_precision))
    print("Recall: {}".format(statistics.weighted_recall))
    print("FScore: {}".format(statistics.weighted_fscore))
    print("Confusion: {}".format(statistics.confusion_matrix))


def test_model(model, criterion, args):
    """
    Test the model performance on the real test set. Use args.model_path to give the path of the model to test.

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
    load_model(model, args.model_path)

    # Create subset otherwise loader is not generated correctly
    test_set = generate_dataset(args.data, 'test')
    test_set = torch.utils.data.Subset(test_set[0], range(len(test_set[0].samples)))
    test_loader = generate_loader(test_set, args.batch_size, args.workers)

    # Resume track model on wandb
    wandb.init(project="RotationNet", name="RotationNet", config=args, resume=True)

    val_statistics = validate(test_loader, model, criterion, args)
    log_summary(val_statistics, "test", test_loader.dataset.dataset.classes)


def validate(test_loader, model, criterion, args):
    """
    Validate the performance on the given data loader using model.eval and without gradient computation.

    Parameters
    ----------
    test_loader : Data loader with test/validation data
    model : RotaitonNet model
    criterion : Pytorch criterion (CrossEntropy for RotationNet)

    Returns
    -------
    Statistics class which contains all the validation statistics such a precision, accuracy, etc.
    """
    statistics = Statistics()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_val, target_val) in enumerate(test_loader):
            loss, (prec1, prec5), y_pred, y_true = execute_batch(model, criterion, input_val,
                                                                 target_val, args)

            statistics.update(loss.data.cpu().numpy(), prec1, prec5, y_pred, y_true)

    return statistics
