import torch.nn as nn
import torchvision.models as models


class SVCNN(nn.Module):

    def __init__(self, nclasses=40, pretraining=True, cnn_name='vgg11', feature_extraction=False):
        super(SVCNN, self).__init__()

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name

        if self.cnn_name == 'resnet18':
            self.net1 = models.resnet18(pretrained=self.pretraining)
            self.net2 = nn.Linear(512, self.nclasses)
        elif self.cnn_name == 'resnet34':
            self.net1 = models.resnet34(pretrained=self.pretraining)
            self.net2 = nn.Linear(512, self.nclasses)
        elif self.cnn_name == 'resnet50':
            self.net1 = models.resnet50(pretrained=self.pretraining)
            self.net2 = nn.Linear(2048, self.nclasses)
        elif self.cnn_name == 'alexnet':
            self.net_1 = models.alexnet(pretrained=self.pretraining).features
            self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            self.net_2._modules['6'] = nn.Linear(4096, self.nclasses)
        elif self.cnn_name == 'vgg11':
            self.net_1 = models.vgg11(pretrained=self.pretraining).features
            self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            self.net_2._modules['6'] = nn.Linear(4096, self.nclasses)
        elif self.cnn_name == 'vgg16':
            self.net_1 = models.vgg16(pretrained=self.pretraining).features
            self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            self.net_2._modules['6'] = nn.Linear(4096, self.nclasses)

        if feature_extraction:
            # Freeze features weights
            for p in self.net_1.parameters():
                p.requires_grad = False

    def forward(self, x):
        y = self.net_1(x)
        return self.net_2(y.view(y.shape[0], -1))
