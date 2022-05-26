Layerwise Learned Convex CNN

This is code built originally the paper https://arxiv.org/abs/1812.11446. It builds upon this paper by using equivalent convex formulations of convolutional neural networks with average pooling, and uses the Burer-Monteiro factorization to enable layerwise learning.

The current state for CIFAR-10 can be run simply by

```
python cifar_convex.py --burer_monteiro
```

One can modify the number of stages through the --ncnn flag, as well as the epochs per stage, learning rate in each stage (as a list), weight decay in each stage (as a list), and dataset (--data\_set). To run the baseline for Fashion-MNIST, for example, one would run

```
python cifar_convex.py --burer_monteiro --ncnn=3 --lr 0.2 0.05 0.005 --data_set=FMNIST
```
