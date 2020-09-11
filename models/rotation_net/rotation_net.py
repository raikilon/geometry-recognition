import logging
import sys

import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger('RotationNet')


class RotationNet(nn.Module):
    def __init__(self, arch, pretrained, output_size, feature_extraction=False, depth=False):
        super(RotationNet, self).__init__()

        if pretrained:
            logger.info("Using pre-trained model '{}'".format(arch))
            original_model = models.__dict__[arch](pretrained=True)
        else:
            logger.info("Using model '{}'".format(arch))
            original_model = models.__dict__[arch]()

        if arch.startswith('alexnet'):
            self.net_1 = original_model.features
            # if training with depth layer
            if depth:
                self.net_1[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)

            # self.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
            # self.classifier = nn.Sequential(
            #     nn.Dropout(),
            #     nn.Linear(256 * 6 * 6, 4096),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(),
            #     nn.Linear(4096, 4096),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(4096, output_size),
            # )
            self.net_2 = original_model.classifier
            self.net_2[-1] = nn.Linear(4096, output_size)
            self.avgpool = original_model.avgpool
            self.modelName = 'alexnet'
        elif arch.startswith('resnet50'):
            # Everything except the last linear layer
            self.net_1 = nn.Sequential(*list(original_model.children())[:-1])
            self.net_2 = nn.Sequential(
                nn.Linear(2048, output_size)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.net_1 = original_model.features
            self.net_2 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, output_size),
            )
            self.modelName = 'vgg16'
        elif arch.startswith("resnext"):
            self.net_1 = nn.Sequential(*list(original_model.children())[:-1])
            self.net_2 = nn.Sequential(
                nn.Linear(2048, output_size)
            )
            self.modelName = 'resnext'
        elif arch.startswith("mobilenet"):
            self.net_1 = original_model.features
            self.net_2 = original_model.classifier
            self.net_2[1] = nn.Linear(original_model.last_channel, output_size)
            self.modelName = 'mobilenet'
        else:
            logger.error("Finetuning not supported on thi architecture")
            sys.exit(1)

        if feature_extraction:
            logger.info("Using features extraction (features weights frozen)")
            # Freeze features weights
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        f = self.net_1(x)
        # allow variable input image size only for alexnet
        if self.modelName == "alexnet":
            f = self.avgpool(f)

        if self.modelName == 'alexnet' or self.modelName == 'vgg16' or self.modelName == 'resnet' or self.modelName == 'resnext':
            f = f.view(f.size(0), -1)
        else:
            # for mobilenet
            f = nn.functional.adaptive_avg_pool2d(f, 1).reshape(f.shape[0], -1)
        y = self.net_2(f)
        return y
