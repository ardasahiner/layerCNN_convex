import torch
from opt_einsum import contract
import numpy as np
import time


n = 1 # number of samples
k = 1024 # number of conv patches
h = 100 # size of each conv patch
c = 10 # number of classes
P = 128 # number of hyperplane arrangements

x = torch.randn((k, n, h)).cuda()
D = np.random.choice([0, 1], size=((n, k, P)))
D = torch.from_numpy(D).float().cuda()
w = torch.randn((P, k, h, c)).cuda()
y = torch.randn((n, c)).cuda()

nruns = 500

with torch.no_grad():
    w = w.detach()
    yhat = contract('ijk,likm,jil->jm', x, w, D)
    Xv = torch.matmul(x, w)
    yhat_naive = torch.mul(D.unsqueeze(3), Xv.permute(2, 1, 0, 3)).permute(0, 2, 3, 1)
    yhat_naive = torch.sum(yhat_naive, dim=(1, 3))
    print('sanity check, einsum vs matmul', torch.norm(yhat-yhat_naive)/torch.norm(yhat))

forward_time = 0
backward_time = 0
for i in range(nruns):
    w = w.detach()
    w.requires_grad = True
    start= time.time()
    Xv = torch.matmul(x, w)
    yhat = torch.mul(D.unsqueeze(3), Xv.permute(2, 1, 0, 3)).permute(0, 2, 3, 1)
    yhat = torch.sum(yhat, dim=(1, 3))
    forward_time += time.time()-start

    loss = torch.norm(yhat - y)**2
    start = time.time()
    loss.backward()
    backward_time += time.time()-start


print('naive matmul average forward time', forward_time/nruns)
print('naive matmul average backward time', backward_time/nruns)
print('naive matmul maximum memory allocated', torch.cuda.max_memory_allocated())
torch.cuda.reset_max_memory_allocated()

forward_time = 0
backward_time = 0
for i in range(nruns):
    w = w.detach()
    w.requires_grad = True
    start= time.time()
    yhat = torch.einsum('ijk,likm,jil->jm', x, w, D)
    forward_time += time.time()-start

    loss = torch.norm(yhat - y)**2
    start = time.time()
    loss.backward()
    backward_time += time.time()-start


print('naive einsum average forward time', forward_time/nruns)
print('naive einsum average backward time', backward_time/nruns)
print('naive einsum maximum memory allocated', torch.cuda.max_memory_allocated())
torch.cuda.reset_max_memory_allocated()

forward_time = 0
backward_time = 0
for i in range(nruns):
    w = w.detach()
    w.requires_grad = True
    start= time.time()
    yhat = contract('ijk,likm,jil->jm', x, w, D)
    forward_time += time.time()-start

    loss = torch.norm(yhat - y)**2
    start = time.time()
    loss.backward()
    backward_time += time.time()-start


print('opt_einsum average forward time', forward_time/nruns)
print('opt_einsum average backward time', backward_time/nruns)
print('opt_einsum maximum memory allocated', torch.cuda.max_memory_allocated())
torch.cuda.reset_max_memory_allocated()
