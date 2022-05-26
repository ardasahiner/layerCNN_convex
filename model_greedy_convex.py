""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import math
import sys


class custom_cvx_layer(torch.nn.Module):
    def __init__(self, in_planes, planes, in_size=32,
                 kernel_size=3, padding=1, avg_size=16, num_classes=10,
                 bias=False, downsample=False,burer_monteiro=False, burer_dim=1):
        super(custom_cvx_layer, self).__init__()

        self.P = planes

        h = int(kernel_size**2) * in_planes
        self.avg_size = avg_size
        self.h = h
        self.num_classes = num_classes
        self.downsample = downsample
        if downsample:
            self.down = psi(2)

        self.out_size = (in_size+ 2*padding - kernel_size + 1) # assumes a stride and dilation of 1
        self.k_eff = self.out_size//self.avg_size
        self.burer_monteiro = burer_monteiro
        self.burer_dim = burer_dim
        self.bias = bias
        self.kernel_size = kernel_size
        self.padding = padding
        self.in_size = in_size
        self.in_planes = in_planes

        self.aggregate_conv = None
        self.aggregated = False

        self.post_pooling = nn.AvgPool2d(kernel_size=avg_size)

        self.sign_pattern_generator = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=padding, bias=bias)
        
        self.sign_pattern_generator.weight.requires_grad = False
        if bias:
            self.sign_pattern_generator.bias.requires_grad = False


        if self.burer_monteiro:
            self.linear_operator = nn.Conv2d(in_planes, planes*self.burer_dim, 
                    kernel_size=kernel_size, padding=self.padding, bias=bias)
            self.fc = nn.Linear(planes*self.burer_dim*self.k_eff*self.k_eff, num_classes)
            self.transformed_linear_operator =  None

        else:
            self.linear_operator = nn.Conv2d(in_planes, planes*num_classes*self.k_eff*self.k_eff, 
                    kernel_size=kernel_size, padding=self.padding, bias=bias)
            self.bias_operator = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x, transformed_model=False):
        # pre-compute sign patterns and downsampling before computational graph necessary
        with torch.no_grad():
            if self.downsample:
                x = self.down(x)

            sign_patterns = self.sign_pattern_generator(x) > 0 # N x P x h x w
        
        N, C, H, W = x.shape
        # for burer-monteiro, m = burer_dim, otherwise m = c * k_eff * k_eff

        if self.burer_monteiro and not transformed_model:
            X_Z = self.linear_operator(x) # N x P*m x h x w
            DX_Z = (X_Z.reshape((N, self.P, -1, H, W))*sign_patterns.unsqueeze(2)).reshape((N, -1,H, W))
            DX_Z = self.post_pooling(DX_Z) # n x P*m x k_eff x k_eff
            DX_Z = DX_Z.reshape((N, -1))
            return self.fc(DX_Z)
        elif self.burer_monteiro and transformed_model:
            X_Z = self.transformed_linear_operator(x)
            DX_Z = torch.einsum('NPhw, NPMhw -> NMhw', sign_patterns.float(), X_Z.reshape((N, self.P, -1, H, W)))
            DX_Z = self.post_pooling(DX_Z) # n x m x k_eff x k_eff
            DX_Z = DX_Z.reshape((N, self.k_eff*self.k_eff, self.num_classes, self.k_eff*self.k_eff))
            pred_transformed = (torch.einsum('naca -> nc', DX_Z) + self.transformed_bias_operator)
            return pred_transformed
        else:
            X_Z = self.linear_operator(x) # N x P*m x h x w
            DX_Z = torch.einsum('NPhw, NPMhw -> NMhw', sign_patterns.float(), X_Z.reshape((N, self.P, -1, H, W)))
            DX_Z = self.post_pooling(DX_Z) # n x m x k_eff x k_eff
            DX_Z = DX_Z.reshape((N, self.k_eff*self.k_eff, self.num_classes, self.k_eff*self.k_eff))
            return (torch.einsum('naca -> nc', DX_Z) + self.bias_operator)/self.k_eff/self.k_eff/math.sqrt(self.num_classes)
    
    def _aggregate_weights(self):
        self.linear_operator.weight.requires_grad = False
        if self.bias:
            self.linear_operator.bias.requires_grad = False

        self.aggregate_conv = nn.Conv2d(self.in_planes, self.P, kernel_size=self.kernel_size, 
                padding=self.padding, bias=self.bias)
        if not self.burer_monteiro:
            aggregate_v = self.linear_operator.weight.data # P*c*k_eff*k_eff x in_planes x kernel_size x kernel_size
            aggregate_v = aggregate_v.reshape((self.P, -1, self.in_planes*self.kernel_size*self.kernel_size)).permute(0, 2, 1) # P x h x ac
            u, s, v = torch.svd(aggregate_v)
            aggregate_v = torch.squeeze(u[:, :, 0]* torch.sqrt(s[:, 0]).unsqueeze(1)) # P x  h
            aggregate_v = aggregate_v.reshape((self.P, self.in_planes, self.kernel_size, self.kernel_size)) # P x in_planes x kernel_size x kernel_size
            self.aggregate_conv.weight.data = aggregate_v/self.k_eff/self.k_eff/math.sqrt(self.num_classes)

            if self.bias:
                aggregate_bias = self.linear_operator.bias.data # P*c*k_eff*k_eff
                aggregate_bias = aggregate_bias.reshape((self.P, -1))
                aggregate_bias = torch.sqrt(torch.norm(aggregate_bias, dim=1)) # P
                self.aggregate_conv.bias.data = aggregate_bias/self.k_eff/math.sqrt(self.num_classes)
        else:
            self.fc.weight.requires_grad = False
            self.fc.bias.requires_grad = False
            self.aggregate_conv.weight.data = self.linear_operator.weight.data
            if self.bias:
                self.aggregate_conv.bias.data = self.linear_operator.bias.data

        self.aggregate_conv.weight.requires_grad = False
        if self.bias:
            self.aggregate_conv.bias.requires_grad = False
        self.aggregated = True

    def forward_next_stage(self, x):
        # pre-compute sign patterns and downsampling before computational graph necessary
        with torch.no_grad():
            if not self.aggregated:
                self._aggregate_weights()
            if self.downsample:
                x = self.down(x)
            sign_patterns = self.sign_pattern_generator(x) > 0 # N x P x h x w
            X_Z = self.aggregate_conv(x) # N x P x h x w
            DX_Z = (sign_patterns * X_Z)

        return DX_Z.detach()
    
class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

    def inverse(self, output):
        """ Expects size (n, channels, block_size_sq, height/block_size, width/block_size)"""
        output = output.reshape((output.shape[0], output.shape[1], self.block_size, self.block_size, output.shape[-2], output.shape[-1]))
        output = output.permute(0, 1, 2, 4, 3, 5)
        input = output.reshape((output.shape[0], output.shape[1], self.block_size*output.shape[3], self.block_size*output.shape[-1]))
        return input.contiguous()


class convexGreedyNet(nn.Module):
    def __init__(self, block, num_blocks, feature_size=256, avg_size=16, num_classes=10, 
                 in_size=32, downsample=[], kernel_size=3, burer_monteiro=False, burer_dim=1,
                 in_planes=3):
        super(convexGreedyNet, self).__init__()
        self.in_planes = feature_size

        self.blocks = []
        self.block = block

        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        
        next_in_planes = self.in_planes

        for n in range(num_blocks):
            pre_factor = 1
            if n in downsample:
                pre_factor *= 4
                avg_size = avg_size // 2
                in_size = in_size // 2
                if n > 2 or (burer_monteiro and burer_dim < 4):
                    next_in_planes = next_in_planes * 2
                self.blocks.append(block(in_planes * pre_factor, next_in_planes, in_size, kernel_size=self.kernel_size,
                                         padding=self.padding, avg_size=avg_size, num_classes=num_classes, bias= n < 1, 
                                         downsample=True, burer_monteiro=burer_monteiro, burer_dim=burer_dim))
            else:
                self.blocks.append(block(in_planes * pre_factor, next_in_planes, in_size, kernel_size=self.kernel_size,
                                         padding=self.padding, avg_size=avg_size, num_classes=num_classes, bias= n < 1, 
                                         downsample=False, burer_monteiro=burer_monteiro, burer_dim=burer_dim))
    
            print(n)
            print(pre_factor*in_planes, next_in_planes)
            in_planes = next_in_planes

        self.blocks = nn.ModuleList(self.blocks)
        for n in range(num_blocks):
            for p in self.blocks[n].parameters():
                p.requires_grad = False

    def unfreezeGradient(self, n):
        for k in range(len(self.blocks)):
            for p in self.blocks[k].parameters():
                p.requires_grad = False
                p.grad = None

        for name, p in self.blocks[n].named_parameters():
            if name.startswith('linear') or name.startswith('fc') or name.startswith('bias'):
                print(name)
                p.requires_grad = True

    def forward(self, a):
        x = a[0]
        N = a[1]
        out = x
        for n in range(N + 1):
            if n < N:
                out = self.blocks[n].forward_next_stage(out).detach()
            else:
                out = self.blocks[n](out)
        return out
