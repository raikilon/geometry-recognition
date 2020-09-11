import os
import torch
import sys
import numpy as np
import time
import wandb
from pytorch3d.transforms import Transform3d
from validate import validate

from util import execute_batch
from util import plot_camera_scene
from util import image_grid
from util import render_shape
import matplotlib.pyplot as plt

# rendering components
from pytorch3d.renderer import (OpenGLPerspectiveCameras, look_at_rotation)

sys.path.insert(0, os.path.abspath('../rotation_net'))
from utils.statistics import Statistics
from utils.statistics import log_data
from utils.model import save_model


def train(model, criterion, optimizer, train_loader, val_loader, args):
    best_prec1 = 0
    epoch_no_improve = 0

    for epoch in range(1000):

        statistics = Statistics()
        model.train()
        start_time = time.time()

        for i, (input, target) in enumerate(train_loader):
            loss, (prec1, prec5), y_pred, y_true = execute_batch(model, criterion, input,
                                                                 target, args.device)

            statistics.update(loss.detach().cpu().numpy(), prec1, prec5, y_pred, y_true)
            # compute gradient and do optimizer step
            optimizer.zero_grad()  #
            loss.backward()
            optimizer.step()

            # if args.net_version == 2:
            #    model.camera_position = model.camera_position.clamp(0, 1)
            del loss
            torch.cuda.empty_cache()

        elapsed_time = time.time() - start_time

        # Evaluate on validation set
        val_statistics = validate(val_loader, model, criterion, args.device)

        log_data(statistics, "train", val_loader.dataset.dataset.classes, epoch)
        log_data(val_statistics, "internal_val", val_loader.dataset.dataset.classes, epoch)

        wandb.log({"Epoch elapsed time": elapsed_time}, step=epoch)
        # print(model.camera_position)
        if epoch % 1 == 0:
            vertices = []
            if args.net_version == 1:
                R = look_at_rotation(model.camera_position, device=args.device)
                T = -torch.bmm(R.transpose(1, 2), model.camera_position[:, :, None])[:, :, 0]
            else:
                t = Transform3d(device=model.device).scale(
                    model.camera_position[3] * model.distance_range).rotate_axis_angle(
                    model.camera_position[0] * model.angle_range, axis="X", degrees=False).rotate_axis_angle(
                    model.camera_position[1] * model.angle_range, axis="Y", degrees=False).rotate_axis_angle(
                    model.camera_position[2] * model.angle_range, axis="Z", degrees=False)

                vertices = t.transform_points(model.vertices)

                R = look_at_rotation(vertices[:model.nviews], device=model.device)
                T = -torch.bmm(R.transpose(1, 2), vertices[:model.nviews, :, None])[:, :, 0]

            cameras = OpenGLPerspectiveCameras(R=R, T=T, device=args.device)
            wandb.log({"Cameras": [wandb.Image(plot_camera_scene(cameras, args.device))]}, step=epoch)
            plt.close()
            images = render_shape(model, R, T, args, vertices)
            wandb.log({"Views": [
                wandb.Image(image_grid(images, rows=int(np.ceil(args.nviews / 2)), cols=2))]},
                step=epoch)
            plt.close()
        #  Save best model and best prediction
        if val_statistics.top1.avg > best_prec1:
            best_prec1 = val_statistics.top1.avg
            save_model("views_net", model, optimizer, args.fname_best)
            epoch_no_improve = 0
        else:
            # Early stopping
            epoch_no_improve += 1
            if epoch_no_improve == 20:
                wandb.run.summary["best_internal_val_top1_accuracy"] = best_prec1
                wandb.run.summary["best_internal_val_top1_accuracy_epoch"] = epoch - 20

                return
