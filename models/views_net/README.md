# PyTorch ViewsNet  
This novel architecture ViewsNet is a fully differentiable model that takes 3D meshes as input and learns the best cameras positions for the classification.

For more information refere to Chapter 4 in my Thesis.

# Execution

To train the model (ViewsNet1) and log the results on W&B execute the following command:

```
main.py DATASET --net_version=2 --nviews=20 --learning_rate_camera=0.001
```

To train the model (ViewsNet2) and log the results on W&B execute the following command:

```
main.py DATASET --net_version=2 --nviews=NVIEWS --learning_rate_camera=0.001
```



