import numpy as np
import wandb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from utils.plotting import generate_confusion_matrix


def log_data(statistics, name, classes, epoch):
    statistics.compute(classes)
    wandb.log({"{}_loss".format(name): statistics.losses.avg,

               "{}_top1_accuracy".format(name): statistics.top1.avg,
               "{}_top5_accuracy".format(name): statistics.top5.avg,

               "{}_macro_recall".format(name): statistics.macro_recall,
               "{}_weighted_recall".format(name): statistics.weighted_recall,

               "{}_macro_fscore".format(name): statistics.macro_fscore,
               "{}_weighted_fscore".format(name): statistics.weighted_recall,

               "{}_macro_precision".format(name): statistics.macro_precision,
               "{}_weighted_precision".format(name): statistics.weighted_precision
               }
              , step=epoch)
    # Send confusion matrix and classification report only every 10 epochs
    # if epoch % 10 == 0:
    #   wandb.log({"{}_confusion_matrix".format(name): generate_confusion_matrix(classes, statistics.confusion_matrix),
    #               "{}_classification_report".format(name): statistics.get_classification_report()
    # "Confusion matrix": wandb.Table(columns=range(args.num_classes),
    #                                data=val_statistics.confusion_matrix.tolist())
    #               },
    #              step=epoch)


def log_summary(statistics, name, classes):
    """
    Log statistics in wandb

    Parameters
    ----------
    statistics : Statistics class
    name : Name of the set where the statistics are computed (e.g. Train, Test, Val)
    classes : List of classes names

    Returns
    -------
    Nothing (everything is send to wandb)
    """

    statistics.compute(classes)
    wandb.run.summary["{}_top5_accuracy".format(name)] = statistics.top5.avg
    wandb.run.summary["{}_top1_accuracy".format(name)] = statistics.top1.avg

    wandb.run.summary["{}_macro_recall".format(name)] = statistics.macro_recall
    wandb.run.summary["{}_weighted_recall".format(name)] = statistics.weighted_recall

    wandb.run.summary["{}_macro_fscore".format(name)] = statistics.macro_fscore
    wandb.run.summary["{}_weighted_fscore".format(name)] = statistics.weighted_recall

    wandb.run.summary["{}_macro_precision".format(name)] = statistics.macro_precision
    wandb.run.summary["{}_weighted_precision".format(name)] = statistics.weighted_precision

    wandb.log({"{}_confusion_matrix".format(name): generate_confusion_matrix(classes, statistics.confusion_matrix),
               "{}_classification_report".format(name): statistics.get_classification_report()})


class Statistics(object):
    """
    Statistics class which contains loss, top1 and top5 accuracies, precision, recall, f score, prediction, targets
    and the confusion matrix.

    Everything is updated autonomously by using the update function.
    """

    def __init__(self):
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.y_true = []
        self.y_pred = []
        self.macro_precision = None
        self.weighted_precision = None
        self.macro_recall = None
        self.weighted_recall = None
        self.macro_fscore = None
        self.weighted_fscore = None
        self.confusion_matrix = None
        self.classification_report = None

        self.classes = []

    def update(self, loss, top1, top5, y_pred, y_true):
        """
        Update statistics values

        Parameters
        ----------
        loss : Model loss
        top1 : Model top1 accuracy
        top5 : Model top5 accuracy
        y_pred : Model predictions
        y_true : Model targets
        y_prob : Model predicted probabilities

        Returns
        -------
        Nothing
        """
        self.losses.update(loss)
        self.top1.update(top1)
        self.top5.update(top5)
        self.y_pred.extend(y_pred)
        self.y_true.extend(y_true)

    def compute(self, classes):
        """
        Compute the confusion matrix, precision, recall and f score with the saved predictions and targets

        Parameters
        ----------
        classes : list of class names

        Returns
        -------
        Nothing
        """
        # self.precision, self.recall, self.fscore, _ = precision_recall_fscore_support(self.y_true, self.y_pred,
        #                                                                              average='weighted')

        self.classification_report = classification_report(self.y_true, self.y_pred, labels=range(len(classes)),
                                                           output_dict=True, zero_division=0)
        self.macro_precision = self.classification_report['macro avg']['precision']
        self.weighted_precision = self.classification_report['weighted avg']['precision']
        self.macro_recall = self.classification_report['macro avg']['recall']
        self.weighted_recall = self.classification_report['weighted avg']['recall']
        self.macro_fscore = self.classification_report['macro avg']['f1-score']
        self.weighted_fscore = self.classification_report['weighted avg']['f1-score']

        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred)

        self.classes = classes

    def get_classification_report(self):
        table = wandb.Table(columns=["class", "accuracy", "precision", "recall", "f1-score", "support"])
        accuracies = self.confusion_matrix.astype("float") / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
        accuracies = accuracies.diagonal()
        for i in range(len(self.classes)):
            table.add_data(self.classes[i], accuracies[i], self.classification_report[str(i)]['precision'],
                           self.classification_report[str(i)]['recall'],
                           self.classification_report[str(i)]['f1-score'],
                           self.classification_report[str(i)]['support'])
        return table


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
