""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy.fftpack import dct, idct
import numpy as np
import math
import sys

def generate_sign_patterns(P, kernel, k_eff, n_channels=3, sparsity=None): 
    # generate sign patterns
    umat = np.random.normal(0, 1, (P, n_channels, kernel, kernel))
    
    if sparsity is not None:
        umat_fft = dct(dct(umat, axis=2), axis=3)
        mask = np.random.choice([1, 0], size=(P, n_channels, kernel, kernel), p=[sparsity, 1-sparsity])

        # make each mask sparsity sparse, try to eliminate non-zero masks
        mask = [m if np.linalg.norm(m) > 0 else np.random.choice([1, 0], size=(n_channels, kernel, kernel), p=[sparsity, 1-sparsity]) for m in mask]
        umat = idct(idct(umat * mask, axis=2), axis=3)
    
    umat = torch.from_numpy(umat).float()
    umat /= kernel**2
    biasmat = torch.from_numpy(np.random.normal(0, torch.mean(torch.std(umat, (2, 3))), (P))).float()
    
    return umat, biasmat

class custom_cvx_layer(torch.nn.Module):
    def __init__(self, in_planes, planes, in_size=32,
                 kernel_size=3, padding=1, avg_size=16, num_classes=10,
                 bias=False, downsample=False, sparsity=None, feat_aggregate='random',
                 nonneg_aggregate=False, burer_monteiro=False, burer_dim=10,
                 sp_weight=None, sp_bias=None, relu=False):
        super(custom_cvx_layer, self).__init__()

        self.P = planes
        self.feat_aggregate = feat_aggregate

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
        self.nonneg_aggregate = nonneg_aggregate

        self.aggregate_conv = None
        self.aggregated = False

        self.post_pooling = nn.AvgPool2d(kernel_size=avg_size)
        self.relu = relu

        if not self.relu:
            self.sign_pattern_generator = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=padding, bias=bias)
            
            if sp_weight is not None:
                u_vectors, bias_vectors = sp_weight, sp_bias
                if self.nonneg_aggregate and in_planes != 3:
                    self.sign_pattern_generator.weight.data = torch.cat((u_vectors, u_vectors), 1)
                else:
                    self.sign_pattern_generator.weight.data = u_vectors
                if bias:
                    if self.nonneg_aggregate and in_planes != 3:
                        self.sign_pattern_generator.bias.data = torch.cat((bias_vectors, bias_vectors), 1)
                    else:
                        self.sign_pattern_generator.bias.data = bias_vectors
    #        else:
    #            u_vectors, bias_vectors = generate_sign_patterns(planes, kernel_size, 
    #                                    self.k_eff, in_planes, sparsity)
            
            self.sign_pattern_generator.weight.requires_grad = False
            if bias:
                self.sign_pattern_generator.bias.requires_grad = False
        else:
            self.act = nn.ReLU()


        if self.burer_monteiro:
            self.linear_operator = nn.Conv2d(in_planes, planes*self.burer_dim, 
                    kernel_size=kernel_size, padding=self.padding, bias=bias)
            self.fc = nn.Linear(planes*self.burer_dim*self.k_eff*self.k_eff, num_classes)
            self.transformed_linear_operator =  None

        else:
            dummy_lin_operator = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=self.padding, bias=bias)
            dummy_fc = nn.Linear(planes*self.k_eff*self.k_eff, num_classes)
            
            U_reshaped = dummy_lin_operator.weight.data.reshape((self.P, -1, self.in_planes*self.kernel_size*self.kernel_size)).permute((0, 2, 1)) # P x h x m
            V_reshaped = dummy_fc.weight.data.t().reshape((self.P, -1, self.k_eff*self.k_eff, self.num_classes)).reshape((self.P, -1, self.k_eff*self.k_eff*self.num_classes)) # P x m x ac
            Z_eff = U_reshaped @ V_reshaped
            Z_eff_conv = Z_eff.permute(0, 2, 1).reshape((-1, self.in_planes, self.kernel_size, self.kernel_size))
            self.linear_operator = nn.Conv2d(in_planes, planes*num_classes*self.k_eff*self.k_eff, 
                    kernel_size=kernel_size, padding=self.padding, bias=bias)
            
            self.linear_operator.weight.data = Z_eff_conv/self.k_eff/self.k_eff/self.num_classes

            self.bias_operator =  nn.Parameter(dummy_fc.bias.data)

            if self.bias:
                U_bias_data = dummy_lin_operator.bias.data.reshape((self.P, -1)) # P x m
                self.linear_operator.bias.data = torch.einsum('Pm, Pma -> Pa', U_bias_data, V_reshaped).reshape(-1) # Pac

            del dummy_lin_operator
            del dummy_fc


    def forward(self, x, transformed_model=False):
        # pre-compute sign patterns and downsampling before computational graph necessary
        with torch.no_grad():
            if self.downsample:
                x = self.down(x)

            if not self.relu:
                sign_patterns = self.sign_pattern_generator(x) > 0 # N x P x h x w
        
        N, C, H, W = x.shape
        # for burer-monteiro, m = burer_dim, otherwise m = c * k_eff * k_eff

        if self.burer_monteiro and not transformed_model:
            X_Z = self.linear_operator(x) # N x P*m x h x w
            if self.relu:
                DX_Z = self.act(X_Z)
            else:
                DX_Z = (X_Z.reshape((N, self.P, -1, H, W))*sign_patterns.unsqueeze(2)).reshape((N, -1,H, W))
            DX_Z = self.post_pooling(DX_Z) # n x P*m x k_eff x k_eff
            DX_Z = DX_Z.reshape((N, -1))
            return self.fc(DX_Z)
        elif self.burer_monteiro and transformed_model:
            X_Z = self.transformed_linear_operator(x)
            if self.relu:
                DX_Z = self.act(X_Z)
            else:
                DX_Z = torch.einsum('NPhw, NPMhw -> NMhw', sign_patterns.float(), X_Z.reshape((N, self.P, -1, H, W)))
            DX_Z = self.post_pooling(DX_Z) # n x m x k_eff x k_eff
            DX_Z = DX_Z.reshape((N, self.k_eff*self.k_eff, self.num_classes, self.k_eff*self.k_eff))
            pred_transformed = (torch.einsum('naca -> nc', DX_Z) + self.transformed_bias_operator)
            return pred_transformed
        else:
            X_Z = self.linear_operator(x) # N x P*m x h x w
            if self.relu:
                DX_Z = self.act(X_Z)
            else:
                DX_Z = torch.einsum('NPhw, NPMhw -> NMhw', sign_patterns.float(), X_Z.reshape((N, self.P, -1, H, W)))
            DX_Z = self.post_pooling(DX_Z) # n x m x k_eff x k_eff
            DX_Z = DX_Z.reshape((N, self.k_eff*self.k_eff, self.num_classes, self.k_eff*self.k_eff))
            return (torch.einsum('naca -> nc', DX_Z) + self.bias_operator)
    
    def generate_Z(self, x=None):
        if not self.burer_monteiro:
            weights_reshaped = self.linear_operator.weight.reshape((self.P, -1, self.in_planes*self.kernel_size*self.kernel_size)).permute((0, 2, 1)) # P x h x ac
            return weights_reshaped
        else:
            if self.transformed_linear_operator is None:
                U_reshaped = self.linear_operator.weight.data.reshape((self.P, -1, self.in_planes*self.kernel_size*self.kernel_size)).permute((0, 2, 1)) # P x h x m
                V_reshaped = self.fc.weight.data.t().reshape((self.P, -1, self.k_eff*self.k_eff, self.num_classes)).reshape((self.P, -1, self.k_eff*self.k_eff*self.num_classes)) # P x m x ac
                Z_eff = U_reshaped @ V_reshaped
                Z_eff_conv = Z_eff.permute(0, 2, 1).reshape((-1, self.in_planes, self.kernel_size, self.kernel_size))
                self.transformed_linear_operator = nn.Conv2d(self.in_planes, self.P*self.num_classes*self.k_eff*self.k_eff,
                    kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)
                self.transformed_linear_operator.weight.data = Z_eff_conv

                # self.transformed_bias_operator = nn.Parameter(torch.zeros(self.num_classes).to(Z_eff_conv.device), requires_grad=False)
                self.transformed_bias_operator =  nn.Parameter(self.fc.bias.data, requires_grad=False)

                if self.bias:
                    U_bias_data = self.linear_operator.bias.data.reshape((self.P, -1)) # P x m
                    self.transformed_linear_operator.bias.data = torch.einsum('Pm, Pma -> Pa', U_bias_data, V_reshaped).reshape(-1) # Pac

                return Z_eff
            else:
                return self.transformed_linear_operator.weight

    def get_Z_grad(self):
        if not self.burer_monteiro:
            return self.linear_operator.weight.grad
        else:
            if self.transformed_linear_operator is None:
                self.generate_Z()
            return self.transformed_linear_operator.weight.grad
        
    def _aggregate_weights(self):
        self.linear_operator.weight.requires_grad = False
        if self.bias:
            self.linear_operator.bias.requires_grad = False

        self.aggregate_conv = nn.Conv2d(self.in_planes, self.P, kernel_size=self.kernel_size, 
                padding=self.padding, bias=self.bias)
        if not self.burer_monteiro:
            if self.feat_aggregate == 'weight_rankone':
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
            if not self.relu:
                sign_patterns = self.sign_pattern_generator(x) > 0 # N x P x h x w
            X_Z = self.aggregate_conv(x) # N x P*m x h x w

            if self.relu:
                DX_Z = self.act(X_Z)
            else:
                DX_Z = sign_patterns.unsqueeze(2) * X_Z.reshape((x.shape[0], self.P, self.burer_dim, x.shape[-2], x.shape[-1]))
                DX_Z = DX_Z.reshape((x.shape[0], -1, x.shape[-2], x.shape[-1]))
                if self.nonneg_aggregate:
                    DX_Z = torch.cat((F.relu(DX_Z), F.relu(-DX_Z)), dim=1)

        return DX_Z.detach()
    
    def nuclear_norm(self):
        weights_reshaped = self.generate_Z()
        return torch.sum(torch.linalg.matrix_norm(weights_reshaped, 'nuc'))

    def constraint_violation(self, x, squared_hinge=False):
        if not self.burer_monteiro:
            return 0
        with torch.no_grad():
            if self.downsample:
                x = self.down(x)

            if not self.relu:
                sign_patterns = 2*(self.sign_pattern_generator(x) > 0)-1 # N x P x h x w
        
        N, C, H, W = x.shape
        X_Z = self.linear_operator(x) # N x P*m x h x w
        DX_Z = (X_Z.reshape((N, self.P, -1, H, W))*sign_patterns.unsqueeze(2))

        if squared_hinge:
            return torch.mean(F.relu(-DX_Z)**2)*self.P
        else:
            return torch.mean(F.relu(-DX_Z))*self.P

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
                 in_size=32, downsample=[], batchnorm=False, sparsity=None, feat_aggregate='random',
                 nonneg_aggregate=False, kernel_size=3, burer_monteiro=False, burer_dim=10,
                 sign_pattern_weights=[], sign_pattern_bias=[], relu=False, in_planes=3):
        super(convexGreedyNet, self).__init__()
        self.in_planes = feature_size

        self.blocks = []
        self.block = block
        self.batchn = batchnorm

        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        
        next_in_planes = self.in_planes

        for n in range(num_blocks):
            sp_weight = sign_pattern_weights[n] if len(sign_pattern_weights) > n else None
            sp_bias = sign_pattern_bias[n] if len(sign_pattern_bias) > n else None
            pre_factor = 1

            if burer_monteiro and n != 0:
                pre_factor *= burer_dim

            if nonneg_aggregate and n != 0:
                pre_factor *= 2

            if n in downsample:
                pre_factor *= 4
                avg_size = avg_size // 2
                in_size = in_size // 2
                if n > 2 or (burer_monteiro and burer_dim < 4):
                    next_in_planes = next_in_planes * 2
                self.blocks.append(block(in_planes * pre_factor, next_in_planes, in_size, kernel_size=self.kernel_size,
                                         padding=self.padding, avg_size=avg_size, num_classes=num_classes, bias= n < 1, 
                                         downsample=True, sparsity=sparsity, feat_aggregate=feat_aggregate,
                                         nonneg_aggregate=nonneg_aggregate, burer_monteiro=burer_monteiro,
                                         burer_dim=burer_dim, sp_weight=sp_weight, sp_bias=sp_bias, relu=relu))
            else:
                self.blocks.append(block(in_planes * pre_factor, next_in_planes, in_size, kernel_size=self.kernel_size,
                                         padding=self.padding, avg_size=avg_size, num_classes=num_classes, bias= n < 1, 
                                         downsample=False, sparsity=sparsity, feat_aggregate=feat_aggregate,
                                         nonneg_aggregate=nonneg_aggregate, burer_monteiro=burer_monteiro,
                                         burer_dim=burer_dim, sp_weight=sp_weight, sp_bias=sp_bias, relu=relu))
    
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

    def unfreezeAll(self):
        for k in range(len(self.blocks)):
            for name, p in self.blocks[k].named_parameters():
                if name.startswith('v'):
                    p.requires_grad = True

    def nuclear_norm(self, n):
        return self.blocks[n].nuclear_norm()

    def hinge_loss(self, a, squared_hinge=False):
        x = a[0]
        N = a[1]
        out = x
        for n in range(N + 1):
            if n < N:
                out = self.blocks[n].forward_next_stage(out).detach()
            else:
                out = self.blocks[n].constraint_violation(out, squared_hinge)
        return out

    def forward(self, a, transformed_model=False):
        x = a[0]
        N = a[1]
        out = x
        for n in range(N + 1):
            if n < N:
                out = self.blocks[n].forward_next_stage(out).detach()
            else:
                out = self.blocks[n](out, transformed_model)
        return out

if __name__ == '__main__':
    torch.manual_seed(9)
    #_ = convexGreedyNet(custom_cvx_layer, 5, feature_size=256, downsample=[2, 3])
    l = custom_cvx_layer(3, 256, in_size=32, burer_monteiro=True, burer_dim=2, avg_size=16, bias=True)
    test_data = torch.randn((10, 3, 32, 32))
    orig_output = l(test_data).detach()
    _ = l.generate_Z(test_data)
    trans_output = l(test_data, True).detach()
    print(torch.norm(trans_output-orig_output))
