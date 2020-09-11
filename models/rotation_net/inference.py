import argparse
import distutils
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.parallel
import torch.optim
import torch.utils.data
from rotation_net import RotationNet
import os
import numpy as np
import shutil
import numpy as np
import torch
import torch.nn.functional
import torch.nn.parallel
import torch.optim
import torch.utils.data
import wandb
import sys
from utils.dataset import generate_dataset
from utils.dataset import generate_loader
from utils.model import load_model
from utils.statistics import Statistics
from utils.statistics import log_summary
from utils.train import execute_batch
from utils.train import execute_inference

parser = argparse.ArgumentParser(description='PyTorch Rotation Training')
# parser.add_argument('data', metavar='PATH', help='path to single object')
parser.add_argument('--num_classes', default=27, type=int, help='Number of classes (default: 27)')
parser.add_argument('--model_path', type=str, help='Model to use')
parser.add_argument('--threshold', default=-50, type=int, help='Inliers threshold')
parser.add_argument('--blender', type=str, help='blender path')
parser.add_argument('--render', type=str, help='render script path')

classes = {10: 'file_cabinet', 0: 'armchair', 23: 'toilet', 18: 'sofa', 2: 'bed', 11: 'floor', 16: 'range_hood',
           12: 'glass_box', 6: 'coffee_table', 15: 'night_stand', 9: 'dresser', 20: 'stool', 24: 'tv_stand',
           8: 'display', 5: 'chair', 26: 'wardrobe', 19: 'stairs', 7: 'desk', 21: 'stove', 25: 'walls', 4: 'bookshelf',
           1: 'bathtub', 13: 'lamp', 17: 'sink', 14: 'mantel', 22: 'table', 3: 'bench'}


def main():
    args = parser.parse_args()
    # REPRODUCIBILITY
    torch.manual_seed(0)
    np.random.seed(0)
    args.vcand = np.load('view_candidates/vcand_case2.npy')
    args.nview = 20

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # Create RotationNet model based on the given architecture.
    # The output size is (num_classes + wrong_view class) * the number of views
    model = RotationNet("alexnet", True, (args.num_classes + 1) * args.nview)

    # Multi GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # Send model to GPU or keep it to the CPU
    model = model.to(device=args.device)
    # switch to evaluate mode
    model.eval()
    model = torch.nn.DataParallel(model)
    load_model(model, args.model_path, 'cpu')

    in_dir = "../../others/separate_meshes/results/non_dirty1"
    out_dir = "../../others/separate_meshes/results/recognition"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.mkdir(out_dir)

    for filename in os.listdir(in_dir):
        if filename.endswith(".obj"):
            args.data = os.path.join(in_dir, filename)

            # delete previous folder
            if os.path.exists('inference_img'):
                shutil.rmtree('inference_img')

            os.mkdir("inference_img")
            os.mkdir("inference_img/train/")
            os.mkdir("inference_img/train/class")

            args.folder = os.path.abspath("inference_img/train/class")

            render_images(args)

            test_set = (generate_dataset("inference_img", 'train', trans=[]),)
            test_set = torch.utils.data.Subset(test_set[0], range(len(test_set[0].samples)))
            test_loader = generate_loader(test_set, args.nview, workers=0)

            with torch.no_grad():
                # input_val, target_val = next(iter(test_loader))
                for batch_idx, (input_val, target_val) in enumerate(test_loader):
                    prediction, score, _, _ = execute_inference(model, input_val, target_val, args)

                    if score[0] < args.threshold:
                        cat = "outlier"
                    else:
                        cat = classes[prediction[0]]
                    input_val = None
                    del input_val
                    target_val = None
                    del target_val
            test_set = None
            del test_set
            test_loader = None
            del test_loader
            print(cat)
            if not os.path.exists(os.path.join(out_dir, cat)):
                os.mkdir(os.path.join(out_dir, cat))
            os.rename(os.path.join(in_dir, filename), os.path.join(out_dir, cat, filename))
            # sys.exit()
        else:
            continue


def render_images(args):
    # white_orthographic
    # print("{} -b {} --python-text 'dataset_creation' -- {} {}".format(args.blender, args.render, args.data, args.folder))
    os.system(
        "{} -b {} --python-text 'dataset_creation' -- {} {}".format(args.blender, args.render, args.data, args.folder))


if __name__ == '__main__':
    main()
