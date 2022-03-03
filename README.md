Layerwise Learned Convex CNN

This is code built originally the paper https://arxiv.org/abs/1812.11446. It builds upon this paper by using equivalent convex formulations of neural networks as proposed originally in Pilanci and Ergen, ICML 2020. This repo is currently a work in progress. 

The current state can be run simply by

```
python cifar_convex.py
```

There are options if you want to save checkpoints after each stage, or load from previous checkpoints. There are options for how different features are aggregated. Also, running the original cifar.py with the --separable tag will allow you to run the non-convex fully separable model which we have convexified. Training the convex program takes about 8.5 hours on one GPU. 

If you would like to use the FFCV data loader, you can use the --ffcv flag to indicate this. Just make sure you have the appropriate .beton files in the correct repositories. One may also use burer-monteiro factorization with the appropriate hidden dimension if desired. 


The current results of the defaults in the codebase are given as follows:

| Stage | Test Accuracy |
|-------|---------------|
| 1     | 69.28         |
| 2     | 72.46         |
| 3     | 73.13         |
| 4     | 73.6          |
| 5     | 72.99         |


Besides tuning feature aggregation from stage to stage, learning rate schedules and exact architectures should be tuned. 
