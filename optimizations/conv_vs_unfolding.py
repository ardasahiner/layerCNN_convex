import torch
import numpy as np
import time
import torch.nn as nn

class psi3(nn.Module):
    def __init__(self, block_size, padding=0):
        super(psi3, self).__init__()
        self.block_size = block_size
        self.padding = padding

    def forward(self, x):
        """Expects x.shape == (batch, channel, height, width).
           Converts to (batch, channel, height / block_size, block_size, 
                                        width / block_size, block_size),
           transposes to put the two 'block_size' dims before channel,
           then reshapes back into (batch, block_size ** 2 * channel, ...)"""

        bs = self.block_size 
        batch, channel, height, width = x.shape
        if ((height % bs) != 0) or (width % bs != 0):
            raise ValueError("height and width must be divisible by block_size")

        kernel_size, stride = height//bs, height//bs
        kernel_size+= 2*self.padding
        #patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)

        unf = nn.Unfold(kernel_size, padding=self.padding, stride=stride)
        patches = unf(x) # n x c*(kernel_size+1)^2 x block_size_sq

        patches = patches.permute(0, 2, 1).contiguous().view(patches.size(0), bs*bs, -1, kernel_size, kernel_size)

        return patches.reshape((patches.size(0), -1, kernel_size, kernel_size))
    
    def inverse(self, output):
        """ Expects size (n, channels, block_size_sq, height/block_size, width/block_size)"""
        output = output.reshape((output.shape[0], output.shape[1], self.block_size, self.block_size, output.shape[-2], output.shape[-1]))
        output = output.permute(0, 1, 2, 4, 3, 5)
        input = output.reshape((output.shape[0], output.shape[1], self.block_size*output.shape[3], self.block_size*output.shape[-1]))
        return input.contiguous()

n = 1 # number of samples
h = 32 # image dimension
ksize = 3 
padding = 1
in_chans = 128
c = 10 # number of classes
P = 128 # number of hyperplane arrangements
avg_size = 4

x = torch.randn((n, in_chans, h, h)).cuda()

umat = torch.randn((P, in_chans, ksize, ksize)).cuda()
biasmat = torch.randn((P)).cuda()

D = (nn.functional.conv2d(x, umat, bias=biasmat, padding=padding) >= 0).int().float()
w = torch.randn((P*c*avg_size*avg_size, in_chans, ksize, ksize)).cuda()
y = torch.randn((n, c)).cuda()

unf = torch.nn.Unfold(kernel_size=ksize, padding=padding)

nruns = 50
downsample_data = psi3(avg_size, padding)
downsample_patterns = psi3(avg_size)

with torch.no_grad():
    x_downsized = downsample_data(x) # n x c_in*avg_size*avg_size x h/avg_size x h/avg_size
    d_downsized = downsample_patterns(D)
    Xv = torch.nn.functional.conv2d(x_downsized, w, groups = avg_size*avg_size) # n x P*c*avg_size*avg_size x h/avg_size x h/avg_size
    Xv0 = torch.nn.functional.conv2d(x, w[:P*c], padding=padding) # n x c*P x h x h
    Xv1 = torch.nn.functional.conv2d(x, w[P*c:2*P*c], padding=padding)

    Xv = Xv.reshape((n, P, c, avg_size*avg_size, h//avg_size, h//avg_size)).permute(0, 2, 1, 3, 4, 5)
    DXv = torch.sum(d_downsized.unsqueeze(1) * Xv.reshape((n, c, P*avg_size*avg_size, h//avg_size, h//avg_size)), dim=(2, 3, 4))

torch.cuda.reset_max_memory_allocated()
forward_time = 0
backward_time = 0
for i in range(nruns):
    w = w.detach()
    w.requires_grad = True
    start= time.time()
    x_downsized = downsample_data(x) # n x c_in*avg_size*avg_size x h/avg_size x h/avg_size
    d_downsized = downsample_patterns(D)
    Xv = torch.nn.functional.conv2d(x_downsized, w, groups = avg_size*avg_size) # n x P*c*avg_size*avg_size x h/avg_size x h/avg_size
    Xv0 = torch.nn.functional.conv2d(x, w[:P*c], padding=padding) # n x c*P x h x h
    Xv1 = torch.nn.functional.conv2d(x, w[P*c:2*P*c], padding=padding)

    Xv = Xv.reshape((n, P, c, avg_size*avg_size, h//avg_size, h//avg_size)).permute(0, 2, 1, 3, 4, 5)
    yhat = torch.sum(d_downsized.unsqueeze(1) * Xv.reshape((n, c, P*avg_size*avg_size, h//avg_size, h//avg_size)), dim=(2, 3, 4))

    forward_time += time.time()-start

    loss = torch.norm(yhat - y)**2
    start = time.time()
    loss.backward()
    backward_time += time.time()-start

print('naive conv2d average forward time', forward_time/nruns)
print('naive conv2d average backward time', backward_time/nruns)
print('naive conv2d maximum memory allocated', torch.cuda.max_memory_allocated()/1e9, 'GB')
torch.cuda.reset_max_memory_allocated()


forward_time = 0
backward_time = 0
for i in range(nruns):
    w = w.detach()
    w.requires_grad = True
    start= time.time()
    x_unf = unf(x).permute(2, 0, 1) # k x n x h
    w_unf = w.reshape((P, c, avg_size*avg_size, in_chans*ksize*ksize)).permute(0, 2, 3, 1) # P x avg_size*avg_size x _ x c -> P x k x h x c
    Xv_unf = torch.matmul(x_unf,  torch.tile(torch.repeat_interleave(w_unf, h//avg_size, dim=1), (1, h//avg_size, 1, 1))) # P x h*h x n x c
    DXv_unf = torch.mul(D.reshape((n, P, -1)).permute(1, 2, 0).unsqueeze(3), Xv_unf)
    yhat = torch.sum(DXv_unf, dim=(0, 1))
    forward_time += time.time()-start

    loss = torch.norm(yhat - y)**2
    start = time.time()
    loss.backward()
    backward_time += time.time()-start

print('unfolded conv2d average forward time', forward_time/nruns)
print('unfolded conv2d average backward time', backward_time/nruns)
print('unfolded conv2d maximum memory allocated', torch.cuda.max_memory_allocated()/1e9, 'GB')
torch.cuda.reset_max_memory_allocated()
