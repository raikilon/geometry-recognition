from options.test_options import TestOptions
from data import DataLoader
from models.mesh_classifier import ClassifierModel
from util.writer import Writer
from data import CreateDataset
import torch


def run_test(dataset=None, epoch=-1, phase="test"):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    if dataset is None:
        dataset = CreateDataset(opt)
        # be consistent with training
        dataset = torch.utils.data.Subset(dataset, range(len(dataset)))
        dataset = DataLoader(dataset, opt)
    else:
        opt.nclasses = len(dataset.dataset.dataset.classes)
        opt.input_nc = dataset.dataset.dataset.opt.input_nc
        dataset.dataset.dataset.opt.num_aug = 1
        # dataset.dataset.dataset.opt.is_train = False
    model = ClassifierModel(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data, epoch)
        loss, (prec1, prec5), y_pred, y_true = model.test()
        writer.update_counter(loss, prec1, prec5, y_pred, y_true)
    if epoch == -1:
        writer.plot_summary("val", dataset.dataset.dataset.classes)
    else:
        writer.plot(epoch, phase, dataset.dataset.dataset.classes)

    return writer.statistics.top1.avg


if __name__ == '__main__':
    run_test()
