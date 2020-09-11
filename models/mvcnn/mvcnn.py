import torch
import torch.nn as nn


class MVCNN(nn.Module):

    def __init__(self, model, num_views=12):
        super(MVCNN, self).__init__()

        self.num_views = num_views

        self.net_1 = model.net_1
        self.net_2 = model.net_2

    def forward(self, x):
        y = self.net_1(x)
        y = y.view(
            (int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))  # (8,12,512,7,7)
        return self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))
