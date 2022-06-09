Layerwise Learned Convex CNN

This is code built originally the paper https://arxiv.org/abs/1812.11446. It builds upon this paper by using equivalent convex formulations of neural networks as proposed originally in Pilanci and Ergen, ICML 2020. This repo is currently a work in progress. 

The latest innovations to the code are to allow for parallel greedy learning for more time-efficient solutions. New methods of aggregation and scaling from layer to layer still need to be explored. For the best-performing fully-convex architecture, run
```
 python cifar_convex_dgl.py --lr 1e-3 1e-5 1e-5 1e-5 1e-5 --ensemble=0
```

This yields the following results

| Stage | Test Accuracy |
|-------|---------------|
| 1     | 56.54         |
| 2     | 61.43         |
| 3     | 69.98         |
| 4     | 75.99         |
| 5     | 77.15         |

If Burer-Monteiro is desired, one can run

```
python cifar_convex_dgl.py --burer_monteiro --lr 0.1 0.01 0.01 0.01 0.01
```
This yields a test accuracy of 80.4. Surprisingly, these models that are trained in parallel do not overfit, so likely better test performance can be extracted by training for longer or with more parameters. 
