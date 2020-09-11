import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import look_at_rotation
from pytorch3d.renderer import PointLights
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, device, renderer, nclasses, args):
        super().__init__()
        self.device = device
        self.renderer = renderer
        self.nclasses = nclasses
        self.nviews = args.nviews
        self.light = args.lights
        self.net_1 = models.alexnet(pretrained=True).features
        self.net_2 = models.alexnet(pretrained=True).classifier
        self.net_2._modules['6'] = nn.Linear(4096, self.nclasses)
        self.lights = PointLights(device=device)
        phi = (1 + np.sqrt(5)) / 2

        vertices = np.array([[1, 1, 1],
                             [1, 1, -1],
                             [1, -1, 1],
                             [1, -1, -1],
                             [-1, 1, 1],
                             [-1, 1, -1],
                             [-1, -1, 1],
                             [-1, -1, -1],
                             [0, 1 / phi, phi],
                             [0, 1 / phi, -phi],
                             [0, -1 / phi, phi],
                             [0, -1 / phi, -phi],
                             [phi, 0, 1 / phi],
                             [phi, 0, -1 / phi],
                             [-phi, 0, 1 / phi],
                             [-phi, 0, -1 / phi],
                             [1 / phi, phi, 0],
                             [-1 / phi, phi, 0],
                             [1 / phi, -phi, 0],
                             [-1 / phi, -phi, 0]])
        vertices = vertices * 3

        # Create an optimizable parameter for the x, y, z position of the camera.
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array(vertices[:self.nviews], dtype=np.float32)).to(device))

    def forward(self, mesh):
        # Render the image using the updated camera position. Based on the new position of the
        # camera we calculate the rotation and translation matrices
        # (1,3)  -> (n,3)
        R = look_at_rotation(self.camera_position, device=self.device)  # (1, 3, 3) -> (n,3,3)
        # (1,3,1) -> (n,3,1)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[:, :, None])[:, :, 0]  # (1, 3) -> (n,3)

        if self.light:
            images = torch.empty(self.nviews * len(mesh), 224, 224, 4, device=self.device)
            # the loop is needed because for now pytorch3d do not allow a batch of lights
            for i in range(self.nviews):
                self.lights.location = self.camera_position[i]
                imgs = self.renderer(meshes_world=mesh.clone(), R=R[None, i], T=T[None, i],
                                     lights=self.lights)
                for k, j in zip(range(len(imgs)), range(0, len(imgs) * self.nviews, self.nviews)):
                    images[i + j] = imgs[k]
        else:
            meshes = mesh.extend(self.nviews)
            # because now we have n elements in R and T we need to expand them to be the same size of meshes
            R = R.repeat(len(mesh), 1, 1)
            T = T.repeat(len(mesh), 1)

            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)

        images = images.permute(0, 3, 1, 2)
        y = self.net_1(images[:, :3, :, :])
        y = y.view((int(images.shape[0] / self.nviews), self.nviews, y.shape[-3], y.shape[-2], y.shape[-1]))
        return self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))

