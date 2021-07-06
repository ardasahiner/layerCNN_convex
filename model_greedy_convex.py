""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy.fftpack import dct, idct
import numpy as np
import math

def generate_sign_patterns(P, kernel, k_eff, bias=True, n_channels=3, sparsity=None): 
    # generate sign patterns
    umat = np.random.normal(0, 1, (P, n_channels, kernel, kernel))
    
    if sparsity is not None:
        umat_fft = dct(dct(umat, axis=2), axis=3)
        mask = np.random.choice([1, 0], size=(P, n_channels, kernel, kernel), p=[sparsity, 1-sparsity])
        umat = idct(idct(umat * mask, axis=2), axis=3)
    
    #umat = umat.transpose(1, 2, 3, 0).reshape((h, P))
    umat = torch.from_numpy(umat).float()
    biasmat = np.random.normal(0, torch.std(umat), (P))
    
    return umat, torch.from_numpy(biasmat).float()

class custom_cvx_layer(torch.nn.Module):
    def __init__(self, in_planes, planes, in_size=32,
                 kernel_size=3, padding=1, avg_size=16, num_classes=10,
                 bias=False, downsample=False, sparsity=None, feat_aggregate='random'):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(custom_cvx_layer, self).__init__()

        self.feat_aggregate = feat_aggregate
        h = int(kernel_size**2) * in_planes
        self.avg_size = avg_size
        self.P = planes
        self.h = h
        self.num_classes = num_classes
        self.downsample = downsample
        if downsample:
            self.down = psi(2)

        self.out_size = (in_size+ 2*padding - kernel_size + 1) # assumes a stride and dilation of 1
        k = int(self.out_size**2)
        self.k_eff= self.out_size//self.avg_size

        # P x k x h x C
        self.v = torch.nn.Parameter(data=torch.zeros(planes*num_classes*self.k_eff*self.k_eff, in_planes, kernel_size, kernel_size)/(kernel_size**2*in_planes*planes), requires_grad=True)
        if bias:
            self.v_bias = torch.nn.Parameter(data=torch.zeros(planes*num_classes*self.k_eff*self.k_eff), requires_grad=True)

        self.bias = bias

        self.u_vectors, self.bias_vectors = generate_sign_patterns(planes, kernel_size, self.k_eff,
                                                                  bias, in_planes, sparsity)


        self.u_vectors = torch.nn.Parameter(data=self.u_vectors, requires_grad=False)
        self.bias_vectors = torch.nn.Parameter(data=self.bias_vectors, requires_grad=False)
        self.kernel_size = kernel_size
        self.padding = padding
        self.unf = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.in_size = in_size
        self.in_planes = in_planes
        self.feat_indices = np.random.choice(np.arange(self.P*num_classes), size=[self.P])
        
        self.downsample_data = psi3(self.k_eff, self.padding)
        self.downsample_patterns = psi3(self.k_eff)
        self.aggregate_v = torch.nn.Parameter(data=torch.zeros(planes*self.k_eff*self.k_eff, in_planes, kernel_size, kernel_size), requires_grad=False)
        self.aggregate_v_bias = torch.nn.Parameter(data=torch.zeros(planes*self.k_eff*self.k_eff), requires_grad=True)
        self.aggregated = False
        
        #classifier_weights = nn.Parameter(data=torch.zeros(num_classes, 1, planes*self.k_eff*self.k_eff), requires_grad=False)

        #for i in range(num_classes):
        #    torch.nn.init.kaiming_uniform_(classifier_weights[i], a=math.sqrt(5))

        ## torch.nn.init.xavier_uniform_(self.v)
        #torch.nn.init.kaiming_uniform_(self.v, mode='fan_out', nonlinearity='relu')
        #fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.v)
        #bound = 1 / math.sqrt(fan_out)
        #torch.nn.init.uniform_(self.v_bias, -bound, bound)

        #with torch.no_grad():
        #    self.v.data = self.v.data*classifier_weights.reshape((num_classes*planes*self.k_eff*self.k_eff, 1, 1, 1))

    def _forward_helper(self, x):
        # first downsample
        if self.downsample:
            x = self.down(x)
            
        x_downsized = self.downsample_data(x) # n x in_planes*k_eff*k_eff x avg_size x avg_size

        with torch.no_grad():
            sign_patterns = (F.conv2d(x, self.u_vectors, bias=self.bias_vectors, padding=self.padding) >= 0).float() # n x P x out_size * out_size

        d_downsized = self.downsample_patterns(sign_patterns) # n x P*k_eff*k_eff x avg_size x avg_size

        # n x P*c*k_eff*k_eff x avg_size x avg_size
        if self.bias:
            Xv_w = torch.nn.functional.conv2d(x_downsized, self.v, bias=self.v_bias, groups = self.k_eff*self.k_eff)
        else:
            Xv_w = torch.nn.functional.conv2d(x_downsized, self.v, groups = self.k_eff*self.k_eff)

        Xv_w = Xv_w.reshape((Xv_w.shape[0], self.k_eff*self.k_eff, self.num_classes, self.P, self.avg_size, self.avg_size)).permute(0, 2, 1, 3, 4, 5)
        Xv_w = Xv_w.reshape((Xv_w.shape[0], self.num_classes, self.P*self.k_eff*self.k_eff, self.avg_size, self.avg_size))
        DXv_w = d_downsized.unsqueeze(1) * Xv_w
        
        return DXv_w

    def forward(self, x):
        DXv_w = self._forward_helper(x) # n x c x P*k_eff*k_eff x avg_size x avg_size
        y_pred = torch.sum(DXv_w, dim=(2, 3, 4))/self.avg_size**2 # N x C
        return y_pred
    
    def _aggregate_weights(self):
        v_norms = torch.sqrt(torch.sum(self.v**2, dim=(1, 2, 3), keepdim=True)) # P*c*k_eff*k_eff x 1 x 1 x 1
        self.v /= (torch.sqrt(v_norms) + 1e-8)
        self.v_bias /= (torch.sqrt(torch.squeeze(v_norms)) + 1e-8)

        if self.feat_aggregate == 'weight_max':
            aggregate_v = torch.max(self.v.reshape((self.k_eff*self.k_eff, self.num_classes. self.P, -1, self.kernel_size, self.kernel_size)), dim=1, keepdim=False)[0]
            self.aggregate_v.data = aggregate_v.reshape((self.P*self.k_eff*self.k_eff, -1, self.kernel_size, self.kernel_size))
            aggregate_v_bias = torch.max(self.v_bias.reshape((self.k_eff*self.k_eff, self.num_classes, self.P, -1)), dim=1, keepdim=False)[0]
            self.aggregate_v_bias.data = aggregate_v_bias.reshape((self.P*self.k_eff*self.k_eff))
        elif self.feat_aggregate == 'weight_rankone':
            aggregate_v = self.v.reshape((self.k_eff*self.k_eff, self.num_classes, self.P, -1, self.kernel_size, self.kernel_size)).permute(0, 2, 1, 3, 4, 5)
            aggregate_v = self.v.reshape((self.P*self.k_eff*self.k_eff, self.num_classes, -1)).permute(0, 2, 1) # P*k^2 x spatial dims x c
            u, s, v = torch.svd(aggregate_v)
            aggregate_v = torch.squeeze(u[:, :, 0] *s[:, 0].unsqueeze(1))
            self.aggregate_v.data = aggregate_v.reshape((self.P*self.k_eff*self.k_eff, -1, self.kernel_size, self.kernel_size))
            aggregate_v_bias = torch.max(self.v_bias.reshape((self.k_eff*self.k_eff, self.num_classes, self.P, -1)), dim=1, keepdim=False)[0]
            self.aggregate_v_bias.data = aggregate_v_bias.reshape((self.P*self.k_eff*self.k_eff))

        self.aggregated = True

    def _forward_aggregated(self, x):
        # first downsample
        if self.downsample:
            x = self.down(x)
            
        x_downsized = self.downsample_data(x) # n x in_planes*k_eff*k_eff x avg_size x avg_size

        with torch.no_grad():
            sign_patterns = (F.conv2d(x, self.u_vectors, bias=self.bias_vectors, padding=self.padding) >= 0).float() # n x P x out_size * out_size

        d_downsized = self.downsample_patterns(sign_patterns) # n x P*k_eff*k_eff x avg_size x avg_size

        # n x P*c*k_eff*k_eff x avg_size x avg_size
        if self.bias:
            Xv_w = torch.nn.functional.conv2d(x_downsized, self.aggregate_v, bias=self.aggregate_v_bias, groups = self.k_eff*self.k_eff)
        else:
            Xv_w = torch.nn.functional.conv2d(x_downsized, self.aggregate_v, groups = self.k_eff*self.k_eff)

        DXv_w = d_downsized * Xv_w
        next_representation = self.downsample_data.inverse(DXv_w.reshape((DXv_w.shape[0],-1, self.P, self.avg_size, self.avg_size)).permute(0, 2, 1, 3, 4))
        
        return next_representation


    def forward_next_stage(self, x):
        next_representation = None
        with torch.no_grad():
            if not self.aggregated:
                self._aggregate_weights()

            if self.feat_aggregate.startswith('weight'):
                next_representation = self._forward_aggregated(x)
            else:
                DXv_w = self._forward_helper(x)
                next_representation = DXv_w.reshape((DXv_w.shape[0], self.num_classes, self.k_eff*self.k_eff, -1, self.avg_size, self.avg_size)).permute(0, 2, 1, 3, 4, 5).reshape((DXv_w.shape[0], -1, self.k_eff*self.k_eff, self.avg_size, self.avg_size))
                next_representation = self.downsample_data.inverse(next_representation)

                if self.feat_aggregate == 'random':
                    next_representation = next_representation[:, self.feat_indices, :, :]

                if self.feat_aggregate == 'max':
                    next_representation = next_representation.reshape((next_representation.shape[0], self.num_classes, self.P, self.out_size, self.out_size))
                    next_representation = torch.max(torch.abs(next_representation), dim=1, keepdim=False)[0]

        # print(torch.max(next_representation), torch.min(next_representation), torch.mean(next_representation))
        return next_representation

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
        output = output.reshape((output.shape[0], -1, self.block_size, self.block_size, output.shape[-2], output.shape[-1]))
        output = output.permute(0, 1, 2, 4, 3, 5)
        input = output.reshape((output.shape[0], output.shape[1], self.block_size*output.shape[3], self.block_size*output.shape[-1]))
        return input.contiguous()


class convexGreedyNet(nn.Module):
    def __init__(self, block, num_blocks, feature_size=256, avg_size=16, num_classes=10, 
                 in_size=32, downsample=[], batchnorm=False, sparsity=None, feat_aggregate='random'):
        super(convexGreedyNet, self).__init__()
        self.in_planes = feature_size

        self.blocks = []
        self.block = block
        self.batchn = batchnorm
        
        in_planes = 3
        next_in_planes = self.in_planes

        for n in range(num_blocks):
            if n in downsample:
                pre_factor = 4
                avg_size = avg_size // 2
                in_size = in_size // 2
                self.blocks.append(block(in_planes * pre_factor, next_in_planes, in_size, kernel_size=3,
                                         padding=1, avg_size=avg_size, num_classes=num_classes, bias=True, 
                                         downsample=True, sparsity=sparsity, feat_aggregate=feat_aggregate))
                # next_in_planes = next_in_planes * 2
            else:
                self.blocks.append(block(in_planes, next_in_planes, in_size, kernel_size=3,
                                         padding=1, avg_size=avg_size, num_classes=num_classes, bias=True, 
                                         downsample=False, sparsity=sparsity, feat_aggregate=feat_aggregate))
            
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
            if name.startswith('v'):
                p.requires_grad = True

    def unfreezeAll(self):
        for k in range(len(self.blocks)):
            for name, p in self.blocks[k].named_parameters():
                if name.startswith('v'):
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

