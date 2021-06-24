import torch
import torch.nn as nn


class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1) # channels_last
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
        output = output.reshape((output.shape[0], output.shape[1], self.block_size, self.block_size, output.shape[-2], output.shape[-1]))
        output = output.permute(0, 1, 2, 4, 3, 5)
        input = output.reshape((output.shape[0], output.shape[1], self.block_size*output.shape[3], self.block_size*output.shape[-1]))
        return input.contiguous()

down = psi3(2)
test = torch.randn((1, 3, 32, 32))
test_down = down(test)

test_down = test_down.reshape((1, 4, 3, 16, 16)).permute(0, 2, 1, 3, 4)

test_up = down.inverse(test_down)

print(torch.norm(test-test_up))
