""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from functools import partial


class separable_block_conv(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes,downsample=False,batchn=True, num_classes=10, in_size=32, avg_size=16):
        super(separable_block_conv, self).__init__()
        self.downsample = downsample
        if downsample:
            self.down = psi(2)

        self.k_eff = (in_size//avg_size)*(in_size//avg_size)
        self.conv1 = nn.Conv2d(in_planes*self.k_eff, planes*num_classes*self.k_eff, kernel_size=3, stride=1, padding=0, bias=False, groups=self.k_eff)
        if batchn:
            self.bn1 = nn.BatchNorm2d(planes*num_classes)
        else:
            self.bn1 = identity()  # Identity

        self.feature_extract = False
        self.sep = psi3(in_size//avg_size, padding=1)
        self.planes=planes

    def forward(self, x):
        if self.downsample:
            x = self.down(x)

        x = self.sep(x)

        out = F.relu(self.bn1(self.conv1(x))) # n x P*c*k_eff x avg_size x avg_size

        if self.feature_extract:
            out = out.reshape((x.shape[0], self.k_eff, -1, out.shape[-2], out.shape[-1]))
            out = self.sep.inverse(out)
            out = out.reshape((x.shape[0], self.planes, -1, out.shape[-2], out.shape[-1]))
            out = torch.max(out, dim=2, keepdim=False)[0]
            print(torch.max(out), torch.min(out), torch.mean(out))

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
        output = output.reshape((output.shape[0], self.block_size, self.block_size, output.shape[-3], output.shape[-2], output.shape[-1]))
        output = output.permute(0, 3, 1, 4, 2, 5) # (n, channels, block_size, height, block_size, width)
        input = output.reshape((output.shape[0], output.shape[1], self.block_size*output.shape[3], self.block_size*output.shape[-1]))
        return input.contiguous()

class separable_auxillary_classifier(nn.Module):
    def __init__(self,avg_size=16,feature_size=256,input_features=256, in_size=32,num_classes=10,n_lin=0,batchn=True):
        super(separable_auxillary_classifier, self).__init__()
        self.n_lin=n_lin

        feature_size = input_features

        if batchn:
            self.bn = nn.BatchNorm2d(feature_size)
        else:
            self.bn = identity()  # Identity

        self.avg_size=avg_size
        self.classifier_weights = nn.Parameter(data=torch.zeros(num_classes, 1, feature_size*(in_size//avg_size)*(in_size//avg_size)), requires_grad=True)
        self.classifier_biases = nn.Parameter(data=torch.zeros(num_classes, 1, 1), requires_grad=True)
        self.feature_size = feature_size
        self.num_classes = num_classes

        for i in range(num_classes):
            torch.nn.init.kaiming_uniform_(self.classifier_weights[i], a=math.sqrt(5))
            torch.nn.init.uniform_(self.classifier_biases, -1, 1)

    def forward(self, x):
        out = x
        if(self.avg_size>1):
            out = F.avg_pool2d(out, self.avg_size)
        out = self.bn(out)
        out = out.view(out.size(0), -1)

        # c x n x d
        out = out.reshape((out.shape[0], self.feature_size, self.num_classes, -1)).permute(2, 0, 1, 3).reshape((self.num_classes, out.shape[0], -1))
        out = out @ self.classifier_weights.permute(0, 2, 1) + self.classifier_biases # c x n x 1
        return out.squeeze().t()


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, input):
        return input

class separableGreedyNet(nn.Module):
    def __init__(self, block, num_blocks, feature_size=256, downsampling=1, downsample=[], batchnorm=True, num_classes=10):
        super(separableGreedyNet, self).__init__()
        self.in_planes = feature_size
        self.down_sampling = psi(downsampling)
        self.downsample_init = downsampling
        self.blocks = []
        self.block = block
        self.blocks.append(block(3*downsampling*downsampling, self.in_planes, downsample=False, batchn=batchnorm))  # n=0
        self.batchn = batchnorm
        self.feature_extract = False
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
            self.blocks[k].feature_extract=True
            for p in self.blocks[k].parameters():
                p.requires_grad = False

        self.blocks[n].feature_extract = False
        for p in self.blocks[n].parameters():
            p.requires_grad = True

    def unfreezeAll(self):
        for k in range(len(self.blocks)):
            if k < len(self.blocks)-1:
                self.blocks[k].feature_extract=True
            for p in self.blocks[k].parameters():
                p.requires_grad = True

    def add_block(self, in_size, avg_size, downsample=False):
        if downsample:
            pre_factor = 4 # the old block needs this factor 4
            self.blocks.append(
                self.block(self.in_planes * pre_factor, self.in_planes, downsample=True, batchn=self.batchn, in_size=in_size, avg_size=avg_size))
            self.in_planes = self.in_planes
        else:
            self.blocks.append(self.block(self.in_planes, self.in_planes,batchn=self.batchn, in_size=in_size, avg_size=avg_size))

    def forward(self, a):
        x = a[0]
        N = a[1]
        out = x
        if self.downsample_init > 1:
            out = self.down_sampling(x)

        with torch.no_grad():
            for n in range(N):
                out = self.blocks[n](out)
        out = self.blocks[N](out)
        return out

