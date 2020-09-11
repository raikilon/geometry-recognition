import os
import torch
import sys
import numpy as np
import argparse
import torch.nn as nn
import wandb
from datetime import datetime
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from classification_data import ClassificationData

from validate import validate
from util import my_collate
from viewsnet_train import train
# rendering components
from pytorch3d.renderer import (OpenGLPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
                                HardFlatShader)

sys.path.insert(0, os.path.abspath('../rotation_net'))
from utils.statistics import log_summary
from utils.model import load_model

parser = argparse.ArgumentParser(description='PyTorch ViewsNet Training')
parser.add_argument('data', metavar='PATH', help='path to dataset')
parser.add_argument('--nviews', default=20, type=int, help="Number of views")
parser.add_argument('--learning_rate', default=0.00001, type=float, help='initial learning rate (default: 0.01)')
parser.add_argument('--learning_rate_camera', default=0.0001, type=float, help='initial learning rate (default: 0.01)')
parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--net_version', default=3, type=int, help='net version')
parser.add_argument('--faces_per_pixel', default=1, type=int, help='Faces per pixel')
parser.add_argument('--lights', action='store_true', help='Faces per pixel')
parser.add_argument('--batch_size', default=40, type=int, help='Batch size')
parser.add_argument('--max_scale', default=1.5, type=float, help='max scale')


def main():
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    # Set the cuda device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if args.net_version == 1:
        from view_net import Model
    else:
        from view_net2 import Model
    # Initialize an OpenGL perspective camera.
    cameras = OpenGLPerspectiveCameras(device=device)

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
    # edges. Refer to blending.py for more details.
    # blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=224,
        blur_radius=0.0,  # np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=args.faces_per_pixel,  # 100
    )

    # We can add a point light in front of the object.
    # lights = PointLights(device=device)
    # lights = DirectionalLights(device=device, direction=self.camera_position)
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(device=device, cameras=cameras)  # , lights=lights)
    )

    dataset = ClassificationData(args.data, "train", device)

    test_idx, train_idx = dataset.train_test_split(dataset.paths, range(len(dataset)), 20, True)
    val_idx, train_idx = dataset.train_test_split([dataset.paths[i] for i in train_idx], train_idx, 20, True)

    test_set = torch.utils.data.Subset(dataset, test_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    train_set = torch.utils.data.Subset(dataset, train_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size // args.nviews, shuffle=False, num_workers=0,
                              pin_memory=True, drop_last=True, collate_fn=my_collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size // args.nviews, shuffle=False, num_workers=1,
                            pin_memory=True, drop_last=True, collate_fn=my_collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size // args.nviews, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=True, collate_fn=my_collate)

    model = Model(device, phong_renderer, dataset.nclasses, args)
    args.num_classes = dataset.nclasses
    # Multi GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)

    optimizer = Adam([
        {'params': model.camera_position, 'lr': args.learning_rate_camera},
        # {'params': model.light_position, 'lr': args.learning_rate_camera},
        {'params': model.net_1.parameters()},
        {'params': model.net_2.parameters()}
    ], lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss().to(device=device)

    wandb.init(project="views_net", config=args)

    args.fname_best = 'views_net{}_model_best{}.pth.tar'.format(args.nviews,
                                                                datetime.now().strftime("%d_%b_%Y_%H_%M_%S"))
    args.device = device

    args.obj_path = train_set.dataset.paths[0][0]

    train(model, criterion, optimizer, train_loader, val_loader, args)

    load_model(model, args.fname_best)

    val_statistics = validate(test_loader, model, criterion, device)

    log_summary(val_statistics, "val", test_loader.dataset.dataset.classes)


if __name__ == '__main__':
    main()
