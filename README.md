Layerwise Learned Convex CNN

This is code built originally the paper https://arxiv.org/abs/1812.11446. It builds upon this paper by using equivalent convex formulations of neural networks as proposed originally in Pilanci and Ergen, ICML 2020. This repo is currently a work in progress. 

The current state can be run simply by

```
python cifar_convex.py
```

There are options if you want to save checkpoints after each stage, or load from previous checkpoints. There are options for how different features are aggregated. Also, running the original cifar.py with the --separable tag will allow you to run the 
non-convex fully separable model which we have convexified. Training the convex program takes about 15 hours on one GPU. 

## Imagenet
Imagenet experiments for 1-hidden layer use the standalone imagenet_single_layer.py

Imagenet experiments for k=2+ can be run with imagenet.py

Note k in the paper corresponds to nlin in the code


To obtain the results for Imagenet

k=3 
```
python imagenet.py IMAGENER_DIR -j THREADS  --ncnn 8 --nlin 2 

```

k=2 

```
python imagenet.py IMAGENER_DIR -j THREADS --ncnn 8 --nlin 1 

```

k=1 model
```
python imagenet_single_layer.py IMAGENER_DIR -j THREADS  --ncnn 8

```
### VGG-11

The VGG-11 model was trained with a new refactored and more modular codebase different from the codebase used for the above models and is thus run from the standalone directory 
refactored_imagenet/

To train the VGG-11 with k=3

```
python imagenet_greedy.py IMAGENER_DIR -j THREADS --arch vgg11_bn --half --dynamic-loss-scale

```
to train the baseline:

```
python imagenet.py IMAGENER_DIR -j THREADS --arch vgg11_bn --half --dynamic-loss-scale

```

### CIFAR experiments
CIFAR experiments can be reproduced using cifar.py

The CIFAR-10 models can be trained:

k=3 (~91.7) 
```
python cifar.py --ncnn 4 --nlin 2 --feature_size 128 --down [1] --bn 1

```

k=2 (~90.4)

```
python cifar.py --ncnn 4 --nlin 1 --feature_size 128 --down [1] --bn 1

```

k=1 (~88.3) 
```
python cifar.py --ncnn 5 --nlin 0 --feature_size 256 

```

Refactored (similar to imagenet_refactored) to train CIFAR-10 coming soon with some improvements in accuracy. 

Contact: please send questions/comments/issues to eugene.belilovsky@umontreal.ca
