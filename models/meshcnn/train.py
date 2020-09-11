import time
from options.train_options import TrainOptions
from data import DataLoader
from data import CreateDataset
from models.mesh_classifier import ClassifierModel
from util.writer import Writer
from test import run_test
from torch.utils.data import random_split
import wandb
import torch
import numpy as np

if __name__ == '__main__':
    opt = TrainOptions().parse()

    # REPRODUCIBILITY
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = CreateDataset(opt)
    # reduce elements
    idx = dataset.reduce_dataset(range(len(dataset)), opt.elements)
    test_idx, train_idx = dataset.train_test_split([dataset.paths[i] for i in idx], idx, opt.val_size, True)
    val_idx, train_idx = dataset.train_test_split([dataset.paths[i] for i in train_idx], train_idx, opt.val_size, True)

    test_set = torch.utils.data.Subset(dataset, test_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    # train_set = torch.utils.data.Subset(dataset, range(len(dataset)))

    # set obj to log pooling
    opt.export_file = val_set.dataset.paths[val_set.indices[0]][0].split("/")[-1]

    test_set.dataset.opt = opt
    val_set.dataset.opt = opt
    train_set.dataset.opt = opt

    dataset = DataLoader(train_set, opt)
    # create val and test dataset without augmentation
    val_dataset = DataLoader(val_set, opt)
    # val_dataset.dataset.dataset.opt.num_aug = 1
    test_dataset = DataLoader(test_set, opt)
    # test_dataset.dataset.dataset.opt.num_aug = 1

    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)

    model = ClassifierModel(opt)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # writer = Writer(opt)
    # total_steps = 0
    best_prec = 0
    epoch_no_improve = 0
    # Track model in wandb
    wandb.init(project="MeshCNN", config=opt, reinit=True)

    # wandb.watch(model)
    for epoch in range(opt.epoch_count, 2000):
        epoch_start_time = time.time()
        # epoch_iter = 0
        # writer.reset_counter()
        for i, data in enumerate(dataset):
            # total_steps += opt.batch_size
            # epoch_iter += opt.batch_size
            model.set_input(data, epoch)
            model.optimize_parameters()

            # model.test()
            # writer.update_counter(loss, prec1, prec5, y_pred, y_true)

            # if i % opt.save_latest_freq == 0:
            #    print('saving the latest model (epoch %d, total_steps %d)' %
            #          (epoch, total_steps))
            #   model.save_network('latest')

        # if epoch % opt.save_epoch_freq == 0:
        #    print('saving the model at the end of epoch %d, iters %d' %
        #          (epoch, total_steps))
        #    model.save_network('latest')
        #    model.save_network(epoch)
        # writer.plot(epoch, "train", dataset.dataset.dataset.classes)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        wandb.log({"Epoch elapsed time": time.time() - epoch_start_time}, step=epoch)
        model.update_learning_rate()
        model.save_network('latest')
        # need to re run test on train dataset because otherwise results are weird
        run_test(dataset, epoch, "train")
        prec = run_test(val_dataset, epoch, "internal_val")
        # prec = run_test(None, epoch)

        #  Save best model and best prediction
        if prec > best_prec:
            best_prec = prec
            model.save_network('best')
            epoch_no_improve = 0
        else:
            # Early stopping
            epoch_no_improve += 1
            if epoch_no_improve == opt.patience:
                model.load_network("best")
                model.save_network('latest')
                wandb.run.summary["best_internal_val_top1_accuracy"] = best_prec
                wandb.run.summary["best_internal_val_top1_accuracy_epoch"] = epoch - opt.patience
                break

    run_test(test_dataset, -1, "val")
    #run_test()
