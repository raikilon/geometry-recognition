#!/bin/bash -l
#SBATCH --job-name=RotationNet
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err
#SBATCH --partition=multi_gpu
#SBATCH --mem=90000
#SBATCH --exclusive
# -w icsnode41
# your commands
# module load python/3.5.6-DL-Lab
module load cudatoolkit/9.0.176
module load cudnn/5.0.5
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

$HOME/libs/bin/python3.5 -m pip install numpy --user
$HOME/libs/bin/python3.5 -m pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu90/torch_stable.html --user
$HOME/libs/bin/python3.5 -m pip install torchvision==0.3.0 -f https://download.pytorch.org/whl/cu90/torch_stable.html --user
$HOME/libs/bin/python3.5 -m pip install scikit-image --user
$HOME/libs/bin/python3.5 -m pip install tensorboardX --user
$HOME/libs/bin/python3.5 -m pip install wandb --user
$HOME/libs/bin/python3.5 -m pip install sklearn --user
$HOME/libs/bin/python3.5 -m pip install pandas --user
$HOME/libs/bin/python3.5 -m pip install mlxtend==0.16.0 --user


wandb login 4b56d228b23ccb870103a7e77500ae9b74e68520


$HOME/libs/bin/python3.5 train_mvcnn.py ../../../ModelNet40_20 --pretrained
