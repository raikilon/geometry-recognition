# Pytorch MeshCNN 

This is a pytorch implementation of MeshCNN [[Hanocka et al.]](https://bit.ly/meshcnn) [[Project Page]](https://ranahanocka.github.io/MeshCNN/)

# Execution

To train the model and log the results on W&B execute the following command:

```
train.py --dataroot DATASET --lr LR --ncf 32 64 128 256 --pool_res 3000 2250 1500 1000 --elements 50 --name NAME --ninput_edges 3750 --batch_size 16 --gpu_ids 0 --norm group --resblocks 1 --niter_decay 100 --patience 20
```


# Acknowledgments

This code was taken from [MeshCNN](https://github.com/ranahanocka/MeshCNN/).
