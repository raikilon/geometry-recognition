import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import look_at_rotation
import torchvision.models as models
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer import PointLights


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
        self.angle_range = np.pi * 2
        self.distance_range = args.max_scale
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
                             [-1 / phi, -phi, 0]], dtype=np.float32)
        self.vertices = torch.from_numpy(vertices).to(device)

        # Create an optimizable parameter for the x, y, z position of the camera.
        self.camera_position = nn.Parameter(torch.rand(4).to(device))
        # self.light_position = torch.from_numpy(np.array([3, 3, 3])).to(device)

    def forward(self, mesh):
        # R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        # T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)

        t = Transform3d(device=self.device).scale(self.camera_position[3] * self.distance_range).rotate_axis_angle(
            self.camera_position[0] * self.angle_range, axis="X", degrees=False).rotate_axis_angle(
            self.camera_position[1] * self.angle_range, axis="Y", degrees=False).rotate_axis_angle(
            self.camera_position[2] * self.angle_range, axis="Z", degrees=False)
        # translation = Translate(T[0][0], T[0][1], T[0][2], device=self.device)

        # t = Transform3d(matrix=self.camera_position)
        vertices = t.transform_points(self.vertices)

        R = look_at_rotation(vertices[:self.nviews], device=self.device)
        T = -torch.bmm(R.transpose(1, 2), vertices[:self.nviews, :, None])[:, :, 0]

        if self.light:
            images = torch.empty(self.nviews * len(mesh), 224, 224, 4, device=self.device)
            # the loop is needed because for now pytorch3d do not allow a batch of lights
            for i in range(self.nviews):
                self.lights.location = vertices[i]
                imgs = self.renderer(meshes_world=mesh.clone(), R=R[None, i], T=T[None, i],
                                     lights=self.lights)
                for k, j in zip(range(len(imgs)), range(0, len(imgs) * self.nviews, self.nviews)):
                    images[i + j] = imgs[k]
        else:
            # self.lights.location = self.light_position
            meshes = mesh.extend(self.nviews)
            # because now we have n elements in R and T we need to expand them to be the same size of meshes
            R = R.repeat(len(mesh), 1, 1)
            T = T.repeat(len(mesh), 1)

            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)  # , lights=self.lights)

        images = images.permute(0, 3, 1, 2)
        y = self.net_1(images[:, :3, :, :])
        y = y.view((int(images.shape[0] / self.nviews), self.nviews, y.shape[-3], y.shape[-2], y.shape[-1]))
        return self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))
