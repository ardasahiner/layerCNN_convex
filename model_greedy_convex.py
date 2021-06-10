""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy.fftpack import dct, idct
import numpy as np

def generate_sign_patterns(P, kernel, h, k, bias=True, n_channels=3, sparsity=None): 
    # generate sign patterns
    umat = np.random.normal(0, 1, (P, n_channels, kernel, kernel))
    
    if sparsity is not None:
        umat_fft = dct(dct(umat, axis=0), axis=1)
        mask = np.random.choice([1, 0], size=(P, n_channels, kernel, kernel), p=[sparsity, 1-sparsity])
        umat = idct(idct(umat * mask, axis=0), axis=1)
    
    umat = umat.transpose(1, 2, 3, 0).reshape((h, P))
    umat = torch.from_numpy(umat).float()
    biasmat = np.random.normal(0, torch.std(umat), (1, 1, P))
    
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
            in_size = in_size // 2

        self.out_size = (in_size+ 2*padding - kernel_size + 1) # assumes a stride and dilation of 1
        k = int(self.out_size**2)
        if downsample:
            self.down = psi(2)
        self.k_eff= int(k/int(avg_size**2))
        
        # P x k x h x C
        self.v = torch.nn.Parameter(data=torch.zeros(planes, self.k_eff, h, num_classes), requires_grad=True)
        if bias:
            self.v_bias = torch.nn.Parameter(data=torch.zeros(planes, self.k_eff, 1, num_classes), requires_grad=True)

        self.bias = bias

        self.u_vectors, self.bias_vectors = generate_sign_patterns(planes, kernel_size, h, k,
                                                                  bias, in_planes, sparsity)


        self.u_vectors = self.u_vectors.to('cuda')
        self.bias_vectors = self.bias_vectors.to('cuda')
        self.kernel_size = kernel_size
        self.padding = padding
        self.unf = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.in_size = in_size
        self.feat_indices = np.random.choice(np.arange(self.P*num_classes), size=[self.P])

#    def _forward_helper(self, x):
#        # first downsample
#        if self.downsample:
#            x = self.down(x)
#
#        # then flatten to obtain shape N x k x h
#        x = self.unf(x).permute(0, 2, 1)
#
#        with torch.no_grad():
#            sign_patterns = (torch.matmul(x, self.u_vectors.to(x.device)) + self.bias_vectors.to(x.device) >= 0).int()
#            sign_patterns = sign_patterns.unsqueeze(3) # N x k x P x 1
#       
#        # N x k x P x h
#        DX = torch.mul(sign_patterns, x.unsqueeze(2))
#        
#        #Sum over adjacent patches, requires first reshaping to NP x h x imh x imw
#        DX = DX.permute(0, 2, 3, 1) # N x P x h x k
#
#        return DX
    def _forward_helper(self, x):
        # first downsample
        if self.downsample:
            x = self.down(x)

        # then flatten to obtain shape N x k x h
        x = self.unf(x).permute(0, 2, 1)

        with torch.no_grad():
            #sign_patterns = (torch.matmul(x, self.u_vectors.to(x.device)) + self.bias_vectors.to(x.device) >= 0).int()
            sign_patterns = (torch.matmul(x, self.u_vectors) + self.bias_vectors >= 0).int()
            sign_patterns = sign_patterns.unsqueeze(3) # N x k x P x 1

        # P x k x N x C
        Xv_w = torch.matmul(x.permute(1,0,2), torch.tile(torch.repeat_interleave(self.v, self.avg_size,dim=1), (1, self.avg_size, 1, 1)))
        if self.bias:
            Xv_w += torch.tile(torch.repeat_interleave(self.v_bias, self.avg_size, dim=1), (1, self.avg_size, 1, 1))
        
        DXv_w = torch.mul(sign_patterns, Xv_w.permute(2, 1, 0, 3)).permute(0, 2, 3, 1) #  N x P x C x k

        return DXv_w

    def forward(self, x):
#        DX = self._forward_helper(x)
#        N = DX.shape[0]
#        DX = DX.reshape((N*self.P, self.h, self.in_size, self.in_size)) # NP x h x imh x imw
#        
#        DX = torch.nn.functional.avg_pool2d(DX, self.avg_size) # NP x h x sqrt(k_eff) x sqrt(k_eff)
#        DX = DX.reshape((N, self.P, self.h, self.k_eff)) # N x P x h x k_eff
#        
#        # P x k_eff x N x C
#        DXv_w = torch.matmul(DX.permute(1, 3, 0, 2), self.v)
#        if self.bias:
#            DXv_w += self.v_bias

        DXv_w = self._forward_helper(x)

        y_pred = torch.sum(DXv_w, dim=(1, 3))/self.avg_size**2 # N x C
        return y_pred

    def forward_next_stage(self, x):
        next_representation = None
        with torch.no_grad():
            DXv_w = self._forward_helper(x)

            if self.feat_aggregate == 'none':
                # N x PC x imh x imw
                next_representation = DXv_w.reshape((DXv_w.shape[0], self.P*self.num_classes, self.out_size, self.out_size))

            elif self.feat_aggregate == 'random':
                next_representation = DXv_w.reshape((DXv_w.shape[0], self.P*self.num_classes, self.out_size, self.out_size))
                next_representation = next_representation[:, self.feat_indices, :, :]


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


class psi2(nn.Module):
    def __init__(self, block_size):
        super(psi2, self).__init__()
        self.block_size = block_size

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

        # reshape (creates a view)
        x1 = x.reshape(batch, channel, height // bs, bs, width // bs, bs)
        # transpose (also creates a view)
        x2 = x1.permute(0, 3, 5, 1, 2, 4)
        # reshape into new order (must copy and thus makes contiguous)
        x3 = x2.reshape(batch, bs ** 2 * channel, height // bs, width // bs)
        return x3


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
                self.blocks.append(block(in_planes * pre_factor, next_in_planes * 2, in_size, kernel_size=3,
                                         padding=1, avg_size=avg_size, num_classes=num_classes, bias=True, 
                                         downsample=True, sparsity=sparsity, feat_aggregate=feat_aggregate))
                next_in_planes = next_in_planes * 2
                in_size = in_size // 2
                avg_size = avg_size // 2
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

        for p in self.blocks[n].parameters():
            p.requires_grad = True

    def unfreezeAll(self):
        for k in range(len(self.blocks)):
            for p in self.blocks[k].parameters():
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

