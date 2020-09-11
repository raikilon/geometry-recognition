import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class DepthDataset(object):

    def __init__(self, root_dir, data, transform=[]):

        data_dir = os.path.join(root_dir, data)

        # Normalization parameters for ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        depth_transform = transform.copy()
        transform.extend([
            transforms.ToTensor(),
            normalize
            # transforms.Grayscale(num_output_channels=1),
        ])
        normalize = transforms.Normalize(mean=[np.mean([0.485, 0.456, 0.406])],
                                         std=[np.mean([0.229, 0.224, 0.225])])
        depth_transform.extend([
            transforms.ToTensor(),
            normalize
            # transforms.Grayscale(num_output_channels=1),
        ])
        # after normalization go back to pil image convert it to grayscale and go back again to tensor
        # depth_transform = transform.copy()
        # depth_transform.extend([
        #    transforms.transforms.ToPILImage(),
        #    transforms.Grayscale(num_output_channels=1),
        #    transforms.ToTensor()
        # ])

        self.classes = list(sorted(os.listdir(data_dir)))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        samples = []
        depth_samples = []
        indices = []
        for idx, target_class in enumerate(sorted(self.classes)):
            target_dir = os.path.join(data_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if path.lower().endswith("png"):
                        # item = path, idx
                        if "depth" in fname.lower():
                            depth_samples.append(path)
                        else:
                            samples.append(path)
                            depth_samples.append(path)
                            indices.append(idx)

        self.class_to_idx = class_to_idx
        self.samples = list(zip(list(zip(samples, depth_samples)), indices))
        # self.depth_samples = depth_samples
        self.targets = indices
        self.transform = transforms.Compose(transform)
        self.depth_transform = transforms.Compose(depth_transform)
        self.root_dir = data_dir

    def __len__(self):
        """return number of points in our dataset"""

        return len(self.samples)

    def __getitem__(self, idx):

        # if torch.is_tensor(idx):
        #    idx = idx.tolist()
        paths, target = self.samples[idx]
        rgb_path, bw_path = paths

        rgb_img = Image.open(rgb_path)
        rgb_img = rgb_img.convert('RGB')
        bw_img = Image.open(bw_path)
        # bw_img = bw_img.convert('RGB')
        bw_img = bw_img.convert('1')

        if self.transform:
            rgb_img = self.transform(rgb_img)
            bw_img = self.depth_transform(bw_img)

        image = torch.cat((bw_img, rgb_img), 0)

        return image, target
