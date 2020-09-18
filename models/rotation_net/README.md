# Pytorch RotationNet

This is a pytorch implementation of RotationNet [Kanezaki et al. 2018](https://arxiv.org/abs/1603.06208) [Project page](https://kanezaki.github.io/rotationnet/)
# Execution

To train the model and log the results on W&B execute the following command:

```
train_rotationnet.py DATASET --pretrained --batch_size 200 --arch=alexnet --optimizer=ADAM --learning_rate=0.00001 --weight_decay=0.0001
```
To train the model on the full dataset and log the results on W&B execute the following command:
```
train_rotationnet.py DATASET --pretrained --num_classes=27 --batch_size=200 --train_type=full --arch=alexnet --optimizer=ADAM --learning_rate=0.00001 --weight_decay=0.0001
```
To evaluate the model on the testset and log the results on W&B execute the following command:
```
train_rotationnet.py DATASET --pretrained --num_classes=27 --model_path=MODELPATH.tar --batch_size=200 --train_type=evaluate --arch=alexnet --optimizer=ADAM --learning_rate=0.00001 --weight_decay=0.0001
```

# Acknowledgments

This code was taken from [RotationNet]()