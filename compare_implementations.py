import torch
from torch import nn
import time

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

P = 256
im_size = 32
avg_size = 16
k_eff = im_size//avg_size
num_classes = 10
in_channels = 15

x = torch.randn((50, in_channels, im_size, im_size)).cuda()
sign_pattern_generator = nn.Conv2d(in_channels, P, kernel_size=3, padding=1, bias=True).cuda()
sign_pattern_generator.weight.requires_grad = False
sign_pattern_generator.bias.requires_grad = False

if avg_size > 1:
    pre_pooling = nn.AvgPool2d(kernel_size=avg_size, padding=0).cuda()
    post_pooling = nn.AvgPool2d(kernel_size=avg_size).cuda()
else:
    pre_pooling = nn.Identity().cuda()
    post_pooling = nn.Identity().cuda()

first_linear_operator = nn.Conv2d(in_channels*P, P*num_classes*k_eff*k_eff, 
        kernel_size=3, stride=3, padding=0, groups=P, bias=False).cuda()

second_linear_operator = nn.Conv2d(in_channels, P*num_classes*k_eff*k_eff,
                            kernel_size=3, padding=1, bias=False).cuda()

second_linear_operator.weight.data = first_linear_operator.weight.data
# second_linear_operator.bias.data = first_linear_operator.bias.data

unfold = nn.Unfold(kernel_size=3, padding=1)
fold = nn.Fold(output_size=k_eff*3, kernel_size=3, padding=0, stride=3)

def down_and_compute_sign_patterns(x):
    N, C, H, W = x.shape
    # pre-compute sign patterns and downsampling before computational graph necessary
    with torch.no_grad():
        sign_patterns = sign_pattern_generator(x) > 0 # N x P x h x w
        x_patches = unfold(x).permute(0, 2, 1) # N x K x c * kernel_size * kernel_size
        sign_patches = sign_patterns.reshape((N, P, -1)).permute(0, 2, 1)  # N x K  x P

        DX_patches = sign_patches.unsqueeze(3) * x_patches.unsqueeze(2) # N x K x P x c*kernel_size*kernel_size
        DX_patches = DX_patches.reshape((N, H, W, -1)).permute(0, 3, 1, 2)
        DX_patches_avg = pre_pooling(DX_patches) # N x P*c*kernel_size x k_eff x k_eff
        DX_patches = DX_patches_avg.reshape((N, -1, k_eff*k_eff))

        DX = fold(DX_patches) # N x P*c, kernel_size*H, kernel_size*W

    return DX

def forward_v2(x):
    N, C, H, W = x.shape
    # pre-compute sign patterns and downsampling before computational graph necessary
    with torch.no_grad():
        DX = down_and_compute_sign_patterns(x)
    
    DX_Z = first_linear_operator(DX) # N x P*m x k_eff x k_eff

    DX_Z = DX_Z.reshape((N, k_eff*k_eff, num_classes, P, k_eff*k_eff))
    return torch.einsum('nacPa -> nc', DX_Z)

def forward_v3(x):
    N, C, H, W = x.shape
    # pre-compute sign patterns and downsampling before computational graph necessary
    with torch.no_grad():
        sign_patterns = sign_pattern_generator(x) > 0 # N x P x h x w
    
    X_Z = second_linear_operator(x) # N x P*m x h x w
    DX_Z = sign_patterns.unsqueeze(2) * X_Z.reshape((N, P,-1, H, W)) # N x m x P x h x w
    DX_Z = DX_Z.reshape((N, -1, H, W))
    DX_Z = post_pooling(DX_Z) # n x P*m x k_eff x k_eff

    DX_Z = DX_Z.reshape((N, k_eff*k_eff, num_classes, P, k_eff*k_eff))
    return torch.einsum('nacPa -> nc', DX_Z)

start1 = time.time()
out1 = forward_v2(x)
out1.backward(torch.randn_like(out1))
torch.cuda.synchronize()
print('first time', time.time()-start1)
print('used memory', torch.cuda.max_memory_allocated()/1e9)
out1.detach()
for param in first_linear_operator.parameters():
    param.grad = None

start2 = time.time()
out2 = forward_v3(x)
out2.backward(torch.randn_like(out2))
torch.cuda.synchronize()
print('second time', time.time()-start2)
print('used memory', torch.cuda.max_memory_allocated()/1e9)

print(torch.norm(out1-out2))
