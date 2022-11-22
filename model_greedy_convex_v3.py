""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy.fftpack import dct, idct
import numpy as np
import math
import sys

class SignPatternGenerator(torch.nn.Module):
    """
        Generates sign patterns for a convex NN module. n is the number of hidden layers in the network,
        i.e. the number of linear hyperplanes which are used to segment the data in conjunction with each other.
    """
    def __init__(self,
                 in_planes,
                 planes,
                 kernel_size,
                 padding,
                 bias,
                 n=1,
                 two_sided=False,
                 tiered=True,
                 use_bn=True,
                 bn_before_layer=True,
                 previous_layer_weights=[],
                 previous_layer_biases=[],
        ):
        """
            two_sided: bool which determines whether one should use the positive and
            negative side of each hyperplane, or just the positive side. The default
            previous approach was to use two-sided=False.
            tiered: bool which determines whether sign patterns for the next hidden layer
            should depend on the hidden activations of the previous layer, or just be linear
            hyperplanes.
        """
        super(SignPatternGenerator, self).__init__()
        self.planes = planes
        self.tiered = tiered

        if self.tiered:
            self.layer_n_generators = []
            curr_in_planes = in_planes
            for layer in range(n):


                curr_layer = [nn.Conv2d(curr_in_planes, planes, kernel_size=kernel_size, padding=padding, bias=bias)]
                if len(previous_layer_weights) > layer:
                    curr_layer[0].weight.data = previous_layer_weights[layer]
                    curr_layer[0].bias.data = previous_layer_biases[layer]
                if use_bn:
                    if bn_before_layer:
                        curr_layer = [nn.BatchNorm2d(curr_in_planes, affine=False)] + curr_layer
                    else:
                        curr_layer += [nn.BatchNorm2d(planes, affine=False)]

                curr_layer = nn.Sequential(*curr_layer)
                self.layer_n_generators.append(curr_layer)
                curr_in_planes = planes
            self.layer_n_generators = nn.ModuleList(self.layer_n_generators)
        else:
            self.layer_n_generators = nn.Conv2d(in_planes, n*planes, kernel_size=kernel_size, padding=padding, bias=bias)

        self.n = n
        self.two_sided = two_sided
        if self.two_sided:
            self.num_patterns = int(2**n)
        else:
            self.num_patterns = 1


    def forward(self, x):
        if self.tiered:
            layer_n_outputs = []
            for layer in range(self.n):
                x = self.layer_n_generators[layer](x)
                layer_n_outputs.append(x)
                x = torch.nn.functional.relu(x)
            layer_n_outputs = torch.cat(layer_n_outputs, dim=1)
        else:
            layer_n_outputs = self.layer_n_generators(x)
        patterns_all = []

        for i in range(self.n):
            curr_len = len(patterns_all)
            if  curr_len == 0:
                patterns_all = [
                    layer_n_outputs[:, (i)*self.planes:(i+1)*self.planes] >= 0,
                ]
                if self.two_sided:
                    patterns_all.append(layer_n_outputs[:, (i)*self.planes:(i+1)*self.planes] < 0)

            else:
                patterns_all.extend([(layer_n_outputs[:, (i)*self.planes:(i+1)*self.planes] >= 0)*patterns_all[j] for j in range(curr_len)])
                if self.two_sided:
                    patterns_all.extend([(layer_n_outputs[:, (i)*self.planes:(i+1)*self.planes] < 0)*patterns_all[j] for j in range(curr_len)])

        return torch.cat(patterns_all[-self.num_patterns:], dim=1)

class custom_cvx_layer(torch.nn.Module):
    def __init__(self, in_planes, planes, in_size=32,
                 kernel_size=3, padding=1, avg_size=16, num_classes=10,
                 bias=True, downsample=False, sparsity=None, feat_aggregate='random',
                 nonneg_aggregate=False, burer_monteiro=False, burer_dim=1,
                 sp_weight=None, sp_bias=None, relu=False, lambd=1e-10, groups=1,
                 pattern_depth=1):
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
        self.a = self.k_eff * self.k_eff
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
        self.decomposed_conv = None
        self.decomposed = False
        self.lambd = lambd

        self.post_pooling = nn.AvgPool2d(kernel_size=avg_size)
        self.groups = groups
        self.downsample_rep = psi(self.k_eff)
        self.relu = relu

        if not self.relu:
            self.sign_pattern_generator = SignPatternGenerator(in_planes, planes, kernel_size, padding, bias, n=pattern_depth)
            for param in self.sign_pattern_generator.parameters():
                param.requires_grad = False

        self.act = nn.ReLU()

        if self.burer_monteiro:
            self.linear_operator = nn.Conv2d(in_planes, planes*self.burer_dim, 
                    kernel_size=kernel_size, padding=self.padding, bias=bias)
            self.fc = nn.Linear(planes*self.burer_dim*self.a, num_classes)
            self.transformed_linear_operator =  None

        else:
            dummy_lin_operator = nn.Conv2d(in_planes//self.groups, planes, kernel_size=kernel_size, padding=self.padding, bias=bias)
            dummy_fc = nn.Linear(planes*self.a, num_classes)
            
            U_reshaped = dummy_lin_operator.weight.data.reshape((self.P, -1, self.in_planes//self.groups*self.kernel_size*self.kernel_size)).permute((0, 2, 1)) # P x h x m
            V_reshaped = dummy_fc.weight.data.t().reshape((self.P, -1, self.a, self.num_classes)).reshape((self.P, -1, self.a*self.num_classes)) # P x m x ac
            Z_eff = U_reshaped @ V_reshaped
            Z_eff_conv = Z_eff.permute(0, 2, 1).reshape((-1, self.in_planes//self.groups, self.kernel_size, self.kernel_size))
            self.linear_operator = nn.Conv2d(in_planes*self.a, planes*num_classes*self.a, 
                    kernel_size=kernel_size, padding=0, bias=bias, groups=self.groups*self.a)
            
            self.linear_operator.weight.data = Z_eff_conv

            self.bias_operator =  nn.Parameter(dummy_fc.bias.data)

            if self.bias:
                U_bias_data = dummy_lin_operator.bias.data.reshape((self.P, -1)) # P x m
                self.linear_operator.bias.data = torch.einsum('pm, pma -> pa', U_bias_data, V_reshaped).reshape(-1) # Pac

            del dummy_lin_operator
            del dummy_fc

        self.rep = None


    def forward(self, x, transformed_model=False, store_activations=False):
        # pre-compute sign patterns and downsampling before computational graph necessary
        with torch.no_grad():
            if self.downsample:
                x = self.down(x)

            if not self.relu:
                sign_patterns = self.sign_pattern_generator(x) > 0 # N x P x h x w
        
        N, C, H, W = x.shape
        # for burer-monteiro, m = burer_dim, otherwise m = c * k_eff * k_eff

        if self.decomposed:
            X_Z = self.decomposed_conv(x)
            DX_Z = self.act(X_Z)
            DX_Z = self.post_pooling(DX_Z) # n x P*m x k_eff x k_eff
            DX_Z = DX_Z.reshape((N, -1))
            return self.fc(DX_Z[:, :DX_Z.shape[1]//2] - DX_Z[:, DX_Z.shape[1]//2:])

        if self.burer_monteiro and not transformed_model:
            X_Z = self.linear_operator(x) # N x P*m x h x w
            if self.relu:
                DX_Z = self.act(X_Z)
            else:
                DX_Z = (X_Z.reshape((N, self.P, -1, H, W))*sign_patterns.unsqueeze(2)).reshape((N, -1,H, W))

            if store_activations:
                # N x Pa x H x W
                self.rep = DX_Z.detach()  

            DX_Z = self.post_pooling(DX_Z) # n x P*m x k_eff x k_eff
            DX_Z = DX_Z.reshape((N, -1))
            return self.fc(DX_Z)
        elif self.burer_monteiro and transformed_model:
            X_Z = self.transformed_linear_operator(x)
            if self.relu:
                DX_Z = self.act(X_Z)
            else:
                DX_Z = torch.einsum('nphw, npmhw -> nmhw', sign_patterns.float(), X_Z.reshape((N, self.P, -1, H, W)))
            DX_Z = self.post_pooling(DX_Z) # n x m x k_eff x k_eff
            DX_Z = DX_Z.reshape((N, self.a, self.num_classes, self.a))
            pred_transformed = (torch.einsum('naca -> nc', DX_Z) + self.transformed_bias_operator)
            return pred_transformed
        else:
            with torch.no_grad():
                x = self.reshape_data_for_groups(x, self.kernel_size, self.padding)

            X_Z = self.linear_operator(x) # N x P*m x h x w
            N, C, H, W = X_Z.shape
            
            if self.relu:
                DX_Z = self.act(X_Z).reshape((N, self.a*self.P, -1, H, W))
            else:
                sp = self.reshape_data_for_groups(sign_patterns.float(), 1, 0)
                DX_Z = torch.einsum('nphw, npmhw -> npmhw', sp, X_Z.reshape((N, self.a*self.P, -1, H, W)))

            if store_activations:
                with torch.no_grad():
                    rep = DX_Z.detach().reshape((N, self.a, self.P, -1, H, W)).reshape((N, self.a, -1, H, W)).permute(0, 2, 1, 3, 4)
                    rep = self.downsample_rep.inverse(rep)
                    self.rep = rep

            DX_Z = DX_Z.sum(1)
            DX_Z = self.post_pooling(DX_Z).squeeze(2).squeeze(2) # n x c
            return DX_Z + self.bias_operator


            #X_Z = self.linear_operator(x) # N x P*m x h x w
            #if self.relu:
            #    DX_Z = self.act(X_Z)
            #else:
            #    DX_Z = torch.einsum('nphw, npmhw -> nmhw', sign_patterns.float(), X_Z.reshape((N, self.P, -1, H, W)))
            #if store_activations:
            #    # N x ac x H x W
            #    self.rep = DX_Z.detach()

            #DX_Z = self.post_pooling(DX_Z) # n x m x k_eff x k_eff
            #DX_Z = DX_Z.reshape((N, self.a, self.num_classes, self.a))
            #return (torch.einsum('naca -> nc', DX_Z) + self.bias_operator)

    def reshape_data_for_groups(self, x, kernel_size, padding):
        x_padded = torch.nn.functional.pad(x, (padding, padding, padding, padding))
        N, C, H, W = x.shape
        new_H = kernel_size + H // self.k_eff - 1
        new_W = kernel_size + W // self.k_eff - 1
        x_unf = torch.nn.functional.unfold(x_padded, kernel_size=(new_H, new_W), padding=0, stride=(new_H-kernel_size+1, new_W-kernel_size+1)) # N x c*newH*newW x k_eff*k_eff
        x_unf = x_unf.reshape((N, C, -1, self.k_eff*self.k_eff))
        x_unf = x_unf.permute(0, 3, 1, 2).reshape((N, C*self.k_eff*self.k_eff, -1)) # N x c*k_eff*k_eff x new_h*newW
        x_fold = torch.nn.functional.fold(x_unf, output_size=(new_H, new_W), kernel_size=1, padding=0) # N x ac x newH x newW

        return x_fold
    
    def generate_Z(self, x=None):
        if not self.burer_monteiro:
            weights_reshaped = self.linear_operator.weight.reshape((self.P, -1, self.in_planes*self.kernel_size*self.kernel_size)).permute((0, 2, 1)) # P x h x ac
            return weights_reshaped
        else:
            if self.transformed_linear_operator is None:
                U_reshaped = self.linear_operator.weight.data.reshape((self.P, -1, self.in_planes*self.kernel_size*self.kernel_size)).permute((0, 2, 1)) # P x h x m
                V_reshaped = self.fc.weight.data.t().reshape((self.P, -1, self.a, self.num_classes)).reshape((self.P, -1, self.a*self.num_classes)) # P x m x ac
                Z_eff = U_reshaped @ V_reshaped
                Z_eff_conv = Z_eff.permute(0, 2, 1).reshape((-1, self.in_planes, self.kernel_size, self.kernel_size))
                self.transformed_linear_operator = nn.Conv2d(self.in_planes, self.P*self.num_classes*self.a,
                    kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)
                self.transformed_linear_operator.weight.data = Z_eff_conv

                # self.transformed_bias_operator = nn.Parameter(torch.zeros(self.num_classes).to(Z_eff_conv.device), requires_grad=False)
                self.transformed_bias_operator =  nn.Parameter(self.fc.bias.data, requires_grad=False)

                if self.bias:
                    U_bias_data = self.linear_operator.bias.data.reshape((self.P, -1)) # P x m
                    self.transformed_linear_operator.bias.data = torch.einsum('pm, pma -> pa', U_bias_data, V_reshaped).reshape(-1) # Pac

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
            if self.downsample:
                x = self.down(x)

            if not self.relu:
                sign_patterns = self.sign_pattern_generator(x) > 0 # N x P x h x w

            x = self.reshape_data_for_groups(x, self.kernel_size, self.padding)
            X_Z = self.linear_operator(x) # N x P*m x h x w
            N, C, H, W = X_Z.shape
            
            if self.relu:
                DX_Z = self.act(X_Z).reshape((N, self.a*self.P, -1, H, W))
            else:
                sp = self.reshape_data_for_groups(sign_patterns.float(), 1, 0)
                DX_Z = torch.einsum('nphw, npmhw -> npmhw', sp, X_Z.reshape((N, self.a*self.P, -1, H, W)))

            rep = DX_Z.detach().reshape((N, self.a, self.P, -1, H, W)).reshape((N, self.a, -1, H, W)).permute(0, 2, 1, 3, 4)
            rep = self.downsample_rep.inverse(rep)

        return rep
    
    def nuclear_norm(self):
        weights_reshaped = self.generate_Z()
        return torch.sum(torch.linalg.matrix_norm(weights_reshaped, 'nuc'))

    def prepare_decomposition(self):
        if not self.decomposed:
            if not self.aggregated:
                self._aggregate_weights()

            # decompose aggregated convs into two convs--one per layer
            self.decomposed_conv = nn.Conv2d(self.in_planes, 2*self.aggregate_conv.weight.shape[0], kernel_size=self.kernel_size,
                                    padding=self.padding, bias=self.bias).to(self.aggregate_conv.weight.device)
            self.decomposed = True

    def decompose_weights(self, x):
        with torch.no_grad():
            if self.downsample:
                x = self.down(x)
            N, C, H, W = x.shape

            if not self.relu:
                sign_patterns = 2*(self.sign_pattern_generator(x) > 0)-1 # N x P x h x w
            
            X_Z = self.aggregate_conv(x) # N x P*m x h x w
            DX_Z = sign_patterns.unsqueeze(2) * X_Z.reshape((N, self.P, self.burer_dim, H, W))
            DX_Z = DX_Z.reshape((N, -1, H, W))
            regress_term = self.act(-DX_Z)
        
        X_Y = self.decomposed_conv(x)[:, :regress_term.shape[1]]
        DX_Y = sign_patterns.unsqueeze(2) * X_Y.reshape((N, self.P, self.burer_dim, H, W))
        DX_Y = DX_Y.reshape((N, -1, H, W))

        return 1/(N*self.P*self.burer_dim*H*W*2)*torch.norm(self.act(regress_term - DX_Y))**2 + self.lambd/self.P * torch.norm(self.decomposed_conv.weight[:regress_term.shape[1]])

    def complete_decomposition(self):
        self.decomposed_conv.weight.data[self.aggregate_conv.weight.shape[0]:] = self.aggregate_conv.weight.data + self.decomposed_conv.weight.data[:self.aggregate_conv.weight.shape[0]]
        self.decomposed_conv.weight.requires_grad = False
        if self.bias:
            self.decomposed_conv.bias.data[self.aggregate_conv.weight.shape[0]:] = self.aggregate_conv.bias.data + self.decomposed_conv.bias.data[:self.aggregate_conv.weight.shape[0]]
            self.decomposed_conv.bias.requires_grad = False

        print(torch.norm(self.decomposed_conv.weight.data))

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
                 sign_pattern_weights=[], sign_pattern_bias=[], relu=False, in_planes=3, decompose=False,
                 lambd=1e-10, pattern_depth=1):
        super(convexGreedyNet, self).__init__()
        self.in_planes = feature_size

        self.blocks = []
        self.block = block
        self.batchn = batchnorm

        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        
        groups=1
        next_in_planes = self.in_planes

        for n in range(num_blocks):
            sp_weight = sign_pattern_weights[n] if len(sign_pattern_weights) > n else None
            sp_bias = sign_pattern_bias[n] if len(sign_pattern_bias) > n else None
            pre_factor = 1

            if n != 0:
                if burer_monteiro:
                    pre_factor *= burer_dim
                else:
                    in_planes = in_planes*num_classes
    #                in_planes = (in_size//avg_size)*(in_size//avg_size)*num_classes
    #                groups = in_planes
    #            else:
                    groups=num_classes
    #                in_planes = in_planes*num_classes
                if nonneg_aggregate:
                    pre_factor *= 2
                if decompose:
                    pre_factor *= 2


            if n in downsample:
                pre_factor *= 4
                avg_size = avg_size // 2
                in_size = in_size // 2
                #if n > 2 or (burer_monteiro and burer_dim < 4):
                next_in_planes = next_in_planes * 2
                self.blocks.append(block(in_planes * pre_factor, next_in_planes, in_size, kernel_size=self.kernel_size,
                                         padding=self.padding, avg_size=avg_size, num_classes=num_classes, bias=True, 
                                         downsample=True, sparsity=sparsity, feat_aggregate=feat_aggregate,
                                         nonneg_aggregate=nonneg_aggregate, burer_monteiro=burer_monteiro,
                                         burer_dim=burer_dim, sp_weight=sp_weight, sp_bias=sp_bias, relu=relu, lambd=lambd, groups=groups, 
                                         pattern_depth=pattern_depth))
            else:
                self.blocks.append(block(in_planes * pre_factor, next_in_planes, in_size, kernel_size=self.kernel_size,
                                         padding=self.padding, avg_size=avg_size, num_classes=num_classes, bias=True, 
                                         downsample=False, sparsity=sparsity, feat_aggregate=feat_aggregate,
                                         nonneg_aggregate=nonneg_aggregate, burer_monteiro=burer_monteiro,
                                         burer_dim=burer_dim, sp_weight=sp_weight, sp_bias=sp_bias, relu=relu, lambd=lambd, groups=groups,
                                         pattern_depth=pattern_depth))
    
            print(n)
            print(pre_factor*in_planes, next_in_planes)
            in_planes = next_in_planes

        self.blocks = nn.ModuleList(self.blocks)
    #    for n in range(num_blocks):
    #        for p in self.blocks[n].parameters():
    #            p.requires_grad = False

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
                if name.startswith('aggregate') or name.startswith('decomp'):
                    p.requires_grad = True

    def nuclear_norm(self, n):
        return self.blocks[n].nuclear_norm()


    def prepare_decomp(self, n):
        self.blocks[n].prepare_decomposition()
        for k in range(len(self.blocks)):
            for p in self.blocks[k].parameters():
                p.requires_grad = False
                p.grad = None

        for name, p in self.blocks[n].named_parameters():
            if name.startswith('decomp'):
                print(name)
                p.requires_grad = True


    def complete_decomp(self, n):
        self.blocks[n].complete_decomposition()

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

    def forward(self, a, transformed_model=False, decompose=False, store_activations=False):
        x = a[0]
        N = a[1]
        out = x
        for n in range(N + 1):
            if n < N:
                if store_activations:
                    out = self.blocks[n].rep.detach().contiguous()
                else:
                    out = self.blocks[n].forward_next_stage(out).detach()
            else:
                if not decompose:
                    out = self.blocks[n](out, transformed_model, store_activations)
                else:
                    out = self.blocks[n].decompose_weights(out)
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
