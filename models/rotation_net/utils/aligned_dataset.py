import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class AlignedDataset(object):

    def __init__(self, root_dir, data, transform=[]):

        data_dir = os.path.join(root_dir, data)

        # Normalization parameters for ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform.extend([
            transforms.ToTensor(),
            normalize
            # transforms.Grayscale(num_output_channels=1),
        ])

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
                        samples.append(path)
                        if path.lower()[-7:-4] == "001":
                            indices.append((idx, 1))
                        elif path.lower()[-7:-4] == "002":
                            indices.append((idx, 2))
                        elif path.lower()[-7:-4] == "003":
                            indices.append((idx, 3))
                        elif path.lower()[-7:-4] == "004":
                            indices.append((idx, 4))
                        elif path.lower()[-7:-4] == "005":
                            indices.append((idx, 5))
                        elif path.lower()[-7:-4] == "006":
                            indices.append((idx, 6))
                        elif path.lower()[-7:-4] == "007":
                            indices.append((idx, 7))
                        elif path.lower()[-7:-4] == "008":
                            indices.append((idx, 8))
                        elif path.lower()[-7:-4] == "009":
                            indices.append((idx, 9))
                        elif path.lower()[-7:-4] == "010":
                            indices.append((idx, 10))
                        elif path.lower()[-7:-4] == "011":
                            indices.append((idx, 11))
                        elif path.lower()[-7:-4] == "012":
                            indices.append((idx, 12))
                        elif path.lower()[-7:-4] == "013":
                            indices.append((idx, 13))
                        elif path.lower()[-7:-4] == "014":
                            indices.append((idx, 14))
                        elif path.lower()[-7:-4] == "015":
                            indices.append((idx, 15))
                        elif path.lower()[-7:-4] == "016":
                            indices.append((idx, 16))
                        elif path.lower()[-7:-4] == "017":
                            indices.append((idx, 17))
                        elif path.lower()[-7:-4] == "018":
                            indices.append((idx, 18))
                        elif path.lower()[-7:-4] == "019":
                            indices.append((idx, 19))
                        elif path.lower()[-7:-4] == "020":
                            indices.append((idx, 20))

        self.class_to_idx = class_to_idx
        self.samples = list(zip(samples, list(list(zip(*indices))[0])))
        # self.depth_samples = depth_samples
        self.targets = indices
        self.transform = transforms.Compose(transform)
        self.root_dir = data_dir

    def __len__(self):
        """return number of points in our dataset"""

        return len(self.samples)

    def __getitem__(self, idx):

        # if torch.is_tensor(idx):
        #    idx = idx.tolist()
        path, _ = self.samples[idx]
        target = self.targets[idx]
        image = Image.open(path)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(np.array(target, dtype=int))
