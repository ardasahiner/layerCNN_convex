""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy.fftpack import dct, idct


def generate_sign_patterns(A, P, bias=True, n_channels=3, sparsity=None): 
    # generate sign patterns
    n, k, h = A.shape
    
    umat = np.random.normal(0, 1, (h,P))
    kernel = int(np.sqrt(h/n_channels))
    
    if sparsity is not None:
        umat = umat.reshape((n_channels, kernel, kernel, P)).transpose(3, 0, 1,2)
        umat_fft = dct(dct(umat, axis=0), axis=1)
        mask = np.random.choice([1, 0], size=(P, n_channels, kernel, kernel), p=[sparsity, 1-sparsity])
        umat = idct(idct(umat * mask, axis=0), axis=1)
        umat = umat.transpose(1, 2, 3, 0).reshape((h, P))
        
    umat = torch.from_numpy(umat).float()
    biasmat = np.random.normal(0, torch.std(umat), (1,k, P))
    
    return umat, torch.fromm_numpy(biasmat).float()

class custom_cvx_layer(torch.nn.Module):
    def __init__(self, h, k, num_neurons, avg_size=16, num_classes=10, bias=False, downsample=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(custom_cvx_layer, self).__init__()
        
        self.avg_size = avg_size
        if downsample:
            self.down = psi(2)
        self.k_eff= int(k/int(avg_size**2))
        
        # P x k x h x C
        self.v = torch.nn.Parameter(data=torch.zeros(num_neurons, self.k_eff, h, num_classes), requires_grad=True)
        
        self.v_bias = torch.nn.Parameter(data=torch.zeros(num_neurons, self.k_eff, 1, num_classes), requires_grad=bias)

    def forward(self, x, u_vectors, bias_vectors):
        # x is N x k x h

        with torch.no_grad():
            sign_patterns = (torch.matmul(x, u_vectors) + bias_vectors >= 0).int()
            sign_patterns = sign_patterns.unsqueeze(3) # N x k x P x 1
        
        # N x k x P x h
        DX = torch.mul(sign_patterns, x.unsqueeze(2))
        N  = DX.shape[0]
        P = DX.shape[2]
        h = DX.shape[3]
        
        #Sum over adjacent patches, requires first reshaping to NP x h x imh x imw
        imh = int(np.sqrt(DX.shape[1]))
        DX = DX.permute(0, 2, 3, 1) # N x P x h x k
        DX = DX.reshape((N*P, h, imh, imh)) # NP x h x imh x imw
        
        DX = torch.nn.functional.avg_pool2d(DX, self.avg_size) # NP x h x sqrt(k_eff) x sqrt(k_eff)
        DX = DX.reshape((N, P, h, self.k_eff)) # N x P x h x k_eff
        
        # P x k_eff x N x C
        DXv_w = torch.matmul(DX.permute(1, 3, 0, 2), self.v) + self.v_bias
        DXv_w = DXv_w.permute(2, 0, 1, 3) # N x P x k_eff x C
        y_pred = torch.sum(DXv_w, dim=(1, 2)) # N x C
        
        return y_pred


class block_conv(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes,downsample=False,batchn=True):
        super(block_conv, self).__init__()
        self.downsample = downsample
        if downsample:
            self.down = psi(2)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if batchn:
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = identity()  # Identity

    def forward(self, x):
        if self.downsample:
            x = self.down(x)
        out = F.relu(self.bn1(self.conv1(x)))
        return out

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
    def __init__(self, block, num_blocks, feature_size=256, downsampling=1, downsample=[], batchnorm=True):
        super(convexGreedyNet, self).__init__()
        self.in_planes = feature_size
        self.down_sampling = psi(downsampling)
        self.downsample_init = downsampling
        self.conv1 = nn.Conv2d(3 * downsampling * downsampling, self.in_planes, kernel_size=3, stride=1, padding=1,
                               bias=not batchnorm)

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(self.in_planes)
        else:
            self.bn1 = identity()  # Identity
        self.RELU = nn.ReLU()
        self.blocks = []
        self.block = block
        self.blocks.append(nn.Sequential(self.conv1, self.bn1, self.RELU))  # n=0
        self.batchn = batchnorm
        for n in range(num_blocks - 1):
            if n in downsample:
                pre_factor = 4
                self.blocks.append(block(self.in_planes * pre_factor, self.in_planes * 2,downsample=True, batchn=batchnorm))
                self.in_planes = self.in_planes * 2
            else:
                self.blocks.append(block(self.in_planes, self.in_planes,batchn=batchnorm))

        self.blocks = nn.ModuleList(self.blocks)
        for n in range(num_blocks):
            for p in self.blocks[n].parameters():
                p.requires_grad = False

    def unfreezeGradient(self, n):
        for k in range(len(self.blocks)):
            for p in self.blocks[k].parameters():
                p.requires_grad = False

        for p in self.blocks[n].parameters():
            p.requires_grad = True

    def unfreezeAll(self):
        for k in range(len(self.blocks)):
            for p in self.blocks[k].parameters():
                p.requires_grad = True

    def add_block(self, downsample=False):
        if downsample:
            pre_factor = 4 # the old block needs this factor 4
            self.blocks.append(
                self.block(self.in_planes * pre_factor, self.in_planes * 2, downsample=True, batchn=self.batchn))
            self.in_planes = self.in_planes * 2
        else:
            self.blocks.append(self.block(self.in_planes, self.in_planes,batchn=self.batchn))

    def forward(self, a):
        x = a[0]
        N = a[1]
        out = x
        if self.downsample_init > 1:
            out = self.down_sampling(x)
        for n in range(N + 1):
            out = self.blocks[n](out)
        return out

