import os
import sys
import torch

from util import execute_batch

sys.path.insert(0, os.path.abspath('../rotation_net'))
from utils.statistics import Statistics


def validate(test_loader, model, criterion, device):
    statistics = Statistics()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_val, target_val) in enumerate(test_loader):
            loss, (prec1, prec5), y_pred, y_true = execute_batch(model, criterion, input_val,
                                                                 target_val, device)

            statistics.update(loss.data.cpu().numpy(), prec1, prec5, y_pred, y_true)

    return statistics
