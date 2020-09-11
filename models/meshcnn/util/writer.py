import os
import time
import sys

# add rotation net packages
sys.path.insert(0, os.path.abspath('../rotation_net'))
from utils.statistics import Statistics
from utils.statistics import log_data
from utils.statistics import log_summary

try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None


class Writer:
    def __init__(self, opt):
        self.num_classes = opt.nclasses
        self.statistics = Statistics()

    def plot(self, epoch, phase, classes):
        #self.statistics.compute(self.num_classes)
        log_data(self.statistics, phase, classes, epoch)

    def plot_summary(self, phase, classes):
        #self.statistics.compute(self.num_classes)
        log_summary(self.statistics, phase, classes)

    def reset_counter(self):
        """
        counts # of correct examples
        """
        self.statistics = Statistics()

    def update_counter(self, loss, prec1, prec5, y_pred, y_true):
        self.statistics.update(loss, prec1, prec5, y_pred, y_true)




