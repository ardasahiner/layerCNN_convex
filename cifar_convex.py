"Greedy layerwise cifar training"
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import gc
import sys

import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast

import os
import argparse

from model_greedy_convex_v3 import *
from torch.autograd import Variable

from utils import *

from random import randint
import datetime
import json

from typing import List


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', nargs='+', default=[0.1], type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ncnn',  default=5,type=int, help='depth of the CNN')
parser.add_argument('--nepochs',  default=50,type=int, help='number of epochs')
parser.add_argument('--epochdecay',  default=15,type=int, help='number of epochs')
parser.add_argument('--avg_size',  default=16,type=int, help='size of averaging ')
parser.add_argument('--feature_size',  default=256,type=int, help='feature size')
parser.add_argument('--ds-type', default=None, help="type of downsampling. Defaults to old block_conv with psi. Options 'psi', 'stride', 'avgpool', 'maxpool'")
parser.add_argument('--ensemble', default=1,type=int,help='compute ensemble')
parser.add_argument('--name', default='',type=str,help='name')
parser.add_argument('--batch_size', default=128,type=int,help='batch size')
parser.add_argument('--bn', default=0,type=int,help='use batchnorm')
parser.add_argument('--debug', default=0,type=int,help='debug')
parser.add_argument('--debug_parameters', default=0,type=int,help='verification that layers frozen')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 6)')
parser.add_argument('--down', default='[2, 3]', type=str,
                        help='layer at which to downsample')


parser.add_argument('--sparsity', default=0.5, type=float,
                        help='sparsity of hyperplane generating arrangements')
parser.add_argument('--signs_sgd', action='store_true', help='Whether to initialize sign patterns from SGD')
parser.add_argument('--sgd_path', default='.', help='path to loadSGD weights from')
parser.add_argument('--relu', action='store_true', 
                        help='Replace Gated ReLU with ReLU (makes model non-convex and non Burer-Monteiro!). Used for sanity check')

parser.add_argument('--feat_agg', default='weight_rankone', type=str,
                        help='way to aggregate features from layer to layer')
parser.add_argument('--multi_gpu', default=0, type=int,
                        help='use multiple gpus')
parser.add_argument('--gpu', default=None, type=int, help='Which GPU to use')

parser.add_argument('--seed', default=0, help="Fixes the CPU and GPU random seeds to a specified number")
parser.add_argument('--save_dir', '-sd', default='checkpoints/', help='directory to save checkpoints into')
parser.add_argument('--checkpoint_path', '-cp', default='', help='path to checkpoint to load')
parser.add_argument('--deterministic', '-det', action='store_true', help='Deterministic operations for numerical stability')
parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save checkpoints')
parser.add_argument('--optimizer', default='SGD', help='What optimizer to use')
parser.add_argument('--reset_momentum', action='store_true', help='Whether to reset the momentum parameter every epochdecay epochs')

parser.add_argument('--data_dir', default='/mnt/dense/sahiner', help='Dataset directory')

parser.add_argument('--group_norm', action='store_true', help='Whether to use group norm penalty (otherwise, standard weight decay is used)')
parser.add_argument('--wd', nargs='+', default=[5e-4], type=float, help='regularization parameter')
parser.add_argument('--mse', action='store_true', help='Whether to use MSE loss (otherwise, softmax cross-entropy is used)')
parser.add_argument('--hinge_loss', action='store_true', help='Whether to enforce hinge loss')
parser.add_argument('--lambda_hinge_loss', nargs='+', default=[1e-4], type=float, help='Hinge loss enforcement parameter')
parser.add_argument('--squared_hinge', action='store_true', help='Whether to use squared hinge loss')


parser.add_argument('--e2e_epochs', default=0, type=int, help='number of epochs after training layerwise to fine-tune e2e')
parser.add_argument('--nonneg_aggregate', action='store_true')
parser.add_argument('--kernel_size', default=3, type=int, help='kernel size of convolutions')

parser.add_argument('--burer_monteiro', action='store_true', help='Whether to use burer-monteiro factorization')
parser.add_argument('--burer_dim', default =1, type=int, help='dimension of burer monteiro')
parser.add_argument('--check_constraint', action='store_true', help='Whether to check qualification constraint')
parser.add_argument('--check_stationary', action='store_true', help='Whether to check stationarity of convolutional weights')

parser.add_argument('--ffcv', action='store_true', help='Whether to use FFCV loaders')
parser.add_argument('--data_set', default='CIFAR10', choices=['CIFAR10', 'STL10', 'FMNIST', 'IMNET'],
                    type=str, help='Dataset name')
parser.add_argument('--test_cifar101', action='store_true', 
                    help='Whether to also test on CIFAR-10.1 dataset. In order to make this work, clone the CIFAR-10.1 github repo in args.data_dir.')

args = parser.parse_args()
opts = vars(args)

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
args.ensemble = args.ensemble>0
args.bn = args.bn > 0
if args.sparsity == 0:
    args.sparsity = None

if len(args.lr) == 1:
    lr_list = [args.lr[0]]*args.ncnn
else:
    lr_list = args.lr
if len(args.wd) == 1:
    wd_list = [args.wd[0]]*args.ncnn
else:
    wd_list = args.wd
if len(args.lambda_hinge_loss) == 1:
    lambda_hinge_list = [args.lambda_hinge_loss[0]]*args.ncnn
else:
    lambda_hinge_list = args.lambda_hinge_loss

for i in range(args.ncnn):
    if wd_list[i] < 0:
        wd_list[i] = 10**wd_list[i]
    if lr_list[i] < 0:
        lr_list[i] = 10**lr_list[i]
    if lambda_hinge_list[i] < 0:
        lambda_hinge_list[i] = 10**lambda_hinge_list[i]

assert args.bn == False, 'batch norm not yet implemented'
assert args.kernel_size %2 == 1, 'kernel size must be odd'
args.debug_parameters = args.debug_parameters > 0
args.multi_gpu = args.multi_gpu > 0

if args.debug:
    args.nepochs = 1 # we run just one epoch per greedy layer training in debug mode

downsample =  list(np.array(json.loads(args.down)))
in_size=32
mode=0

if args.seed is not None:
    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

save_name = 'convex_layersize_'+str(args.feature_size) +'down_' +  args.down + args.data_set
#logging
time_stamp = str(datetime.datetime.now().isoformat())

name_log_dir = ''.join('{}{}-'.format(key, val) for key, val in sorted(opts.items()))+time_stamp
name_log_dir = 'runs/'+name_log_dir

name_log_txt = time_stamp + save_name + str(randint(0, 1000)) + args.name
debug_log_txt = name_log_txt + '_debug.log'
name_save_model = args.save_dir + name_log_txt
name_log_txt=name_log_txt   +'.log'


with open(name_log_txt, "a") as text_file:
    print(opts, file=text_file)

use_cuda = torch.cuda.is_available()
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
in_planes = 3

if args.data_set == 'CIFAR10' or args.data_set == 'STL10' or args.data_set == 'FMNIST':
    num_classes = 10
elif args.data_set == 'IMNET':
    num_classes = 1000
else:
    assert False, "dataset name not in CIFAR10, STL10, FMNIST, IMNET"

#if args.ffcv:
#    from ffcv.fields import IntField, RGBImageField
#    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
#    from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
#        RandomResizedCropRGBImageDecoder
#    from ffcv.loader import Loader, OrderOption
#    from ffcv.pipeline.operation import Operation
#    from ffcv.transforms import RandomHorizontalFlip, Cutout, NormalizeImage, \
#        RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
#    from ffcv.transforms.common import Squeeze
#    from ffcv.writer import DatasetWriter
#
#    print('using ffcv')
#
#    if args.data_set == 'CIFAR10':
#        CIFAR_MEAN = [125.307, 122.961, 113.8575]
#        CIFAR_STD = [51.5865, 50.847, 51.255]
#        in_size = 32
#
#        paths = {
#                'train': os.path.join(args.data_dir, 'cifar_train.beton'),
#                'test': os.path.join(args.data_dir, 'cifar_test.beton')
#            }
#
#        loaders = {}
#
#        for name in ['train', 'test']:
#            label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
#            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
#            if name == 'train':
#                image_pipeline.extend([
#                    RandomHorizontalFlip(),
#                    RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
#                    Cutout(4, tuple(map(int, CIFAR_MEAN))),
#                ])
#            image_pipeline.extend([
#                ToTensor(),
#                ToDevice('cuda:0', non_blocking=True),
#                ToTorchImage(),
#                Convert(torch.float16),
#                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#            ])
#
#            ordering = OrderOption.QUASI_RANDOM if name == 'train' else OrderOption.SEQUENTIAL
#
#            loaders[name] = Loader(paths[name], batch_size=args.batch_size, num_workers=args.workers,
#                                   order=ordering, drop_last=(name == 'train'), os_cache=True,
#                                   pipelines={'image': image_pipeline, 'label': label_pipeline})
#
#    else:
#        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
#        IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
#        DEFAULT_CROP_RATIO = 224/256
#        paths = {
#                'train': os.path.join(args.data_dir, 'train_500_0.50_90.ffcv'),
#                'test': os.path.join(args.data_dir, 'val_500_0.50_90.ffcv')
#            }
#
#        loaders = {}
#
#        for name in ['train', 'test']:
#
#            res = 224
#            in_size = res
#
#            if name == 'train':
#                decoder = RandomResizedCropRGBImageDecoder((res, res))
#                image_pipeline: List[Operation] = [
#                    decoder,
#                    RandomHorizontalFlip(),
#                    ToTensor(),
#                    ToDevice('cuda:0', non_blocking=True),
#                    ToTorchImage(),
#                    NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
#                ]
#            else:
#                cropper = CenterCropRGBImageDecoder((res, res), ratio=DEFAULT_CROP_RATIO)
#                image_pipeline = [
#                    cropper,
#                    ToTensor(),
#                    ToDevice('cuda:0', non_blocking=True),
#                    ToTorchImage(),
#                    NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
#                ]
#
#
#            label_pipeline: List[Operation] = [
#                IntDecoder(),
#                ToTensor(),
#                Squeeze(),
#                ToDevice('cuda:0', non_blocking=True)
#            ]
#
#            ordering = OrderOption.QUASI_RANDOM if name=='train' else OrderOption.SEQUENTIAL
#            loaders[name] = Loader(paths[name], batch_size=args.batch_size, num_workers=args.workers,
#                                   order=ordering, drop_last=(name == 'train'),
#                                   pipelines={'image': image_pipeline, 'label': label_pipeline}, 
#                                   os_cache=False)
#
#
#    trainloader_classifier = loaders['train']
#    testloader = loaders['test']

#else:
print('not using ffcv')

if args.data_set == 'IMNET':
    assert False, "Need to use FFCV with imagenet"
# Data
print('==> Preparing data..')

if args.data_set == 'CIFAR10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset_class = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True,transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

elif args.data_set == 'STL10':
    in_size = 96
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    trainset_class = torchvision.datasets.STL10(root=args.data_dir, split='train', download=True,transform=transform_train)
    testset = torchvision.datasets.STL10(root=args.data_dir, split='test', download=True, transform=transform_test)


elif args.data_set == 'FMNIST':
    in_planes = 1
    in_size = 28
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    trainset_class = torchvision.datasets.FashionMNIST(root=args.data_dir, train=True, download=True,transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=args.data_dir, train=False, download=True, transform=transform)

else:
    assert False, 'Something with dataset went wrong'

trainloader_classifier = torch.utils.data.DataLoader(trainset_class, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

if args.test_cifar101:
    X = np.load(os.path.join(args.data_dir, 'CIFAR-10.1', 'datasets', 'cifar10.1_v6_data.npy'))
    y = np.load(os.path.join(args.data_dir, 'CIFAR-10.1', 'datasets', 'cifar10.1_v6_labels.npy'))
    X = np.transpose(X, (0, 3, 1, 2))

    testset = PrepareData(X, y)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Model

print('==> Building model..')
n_cnn=args.ncnn
sign_pattern_weights = []
sign_pattern_bias = []

if args.signs_sgd:
    sgd_model = torch.load(args.sgd_path)
    sgd_blocks = sgd_model['net'].module[0] # should be blocks with CNN, BN, ReLU
    sgd_blocks = sgd_blocks.blocks

    for n in range(n_cnn):
        for name, param in sgd_blocks[n].named_parameters():
            if 'weight' in name:
                sign_pattern_weights.append(param)
            elif 'bias' in name:
                sign_pattern_bias.append(param)

net = convexGreedyNet(custom_cvx_layer, n_cnn, args.feature_size, in_size=in_size, avg_size=args.avg_size, num_classes=num_classes,
                      downsample=downsample, batchnorm=args.bn, sparsity=args.sparsity, feat_aggregate=args.feat_agg,
                      nonneg_aggregate=args.nonneg_aggregate, kernel_size=args.kernel_size, 
                      burer_monteiro=args.burer_monteiro, burer_dim=args.burer_dim, sign_pattern_weights=sign_pattern_weights,
                      sign_pattern_bias=sign_pattern_bias, relu=args.relu, in_planes = in_planes)
    
with open(name_log_txt, "a") as text_file:
    print(net, file=text_file)

n_parameters = sum(p.numel() for p in net.parameters())
print('number of params:', n_parameters)

if args.multi_gpu:
    net = torch.nn.DataParallel(net).cuda()
net = net.cuda()
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
if args.deterministic:
    torch.use_deterministic_algorithms(True)

if args.mse:
    criterion_classifier = nn.MSELoss()
else:
    criterion_classifier = nn.CrossEntropyLoss()

def train_classifier(epoch,n):
    print('\nSubepoch: %d' % epoch)
    net.train()
    for k in range(n):
        if args.multi_gpu:
            net.module.blocks[k].eval()
        else:
            net.blocks[k].eval()
    
    if args.debug_parameters:
    #This is used to verify that early layers arent updated
        import copy
        #store all parameters on cpu as numpy array
        net_cpu = copy.deepcopy(net).cpu()
        if args.multi_gpu:
            net_cpu.module.state_dict()
        else:
            net_cpu_dict = net_cpu.state_dict()
        with open(debug_log_txt, "a") as text_file:
            print('n: %d'%n)
            for param in net_cpu_dict.keys():
                net_cpu_dict[param]=net_cpu_dict[param].numpy()
                print("parameter stored on cpu as numpy: %s  "%(param),file=text_file)

    train_loss = 0
    correct = 0
    total = 0
    hinge_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader_classifier):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if args.mse:
            targets_loss = nn.functional.one_hot(targets, num_classes=num_classes).float()
        else:
            targets_loss = targets

        optimizer.zero_grad()

        with autocast():
            outputs = net.forward([inputs,n])
            loss = criterion_classifier(outputs, targets_loss)

            if args.group_norm:
                loss += wd_list[n]*net.nuclear_norm(n)
            if args.hinge_loss:
                curr_hinge_loss = net.hinge_loss([inputs, n], args.squared_hinge)
                loss += lambda_hinge_list[n] * curr_hinge_loss
                hinge_loss += lambda_hinge_list[n] * curr_hinge_loss.item()

            if torch.isnan(loss):
                print('nan loss!')
                sys.exit(-1)

            train_loss += loss.item()
            _, predicted = torch.max(outputs.detach().data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if args.debug_parameters:
            if args.multi_gpu:
                s_dict = net.module.state_dict()
            else:
                s_dict = net.state_dict()
            with open(debug_log_txt, "a") as text_file:
                print("iteration %d" % (batch_idx), file=text_file)
                for param in s_dict.keys():
                    diff = np.sum((s_dict[param].cpu().numpy()-net_cpu_dict[param])**2)
                    print("n: %d parameter: %s size: %s changed by %.5f" % (n,param,net_cpu_dict[param].shape,diff),file=text_file)


        progress_bar(batch_idx, len(trainloader_classifier), 'Loss: %.3f | Acc: %.3f%% (%d/%d) |  Hinge Loss: %.3f'
            % (train_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total,hinge_loss/(batch_idx+1)))

    acc = 100.*float(correct)/float(total)
    return acc, train_loss/(batch_idx+1)


def check_stationarity(n):
    print('\nChecking stationarity for stage: %d' % n)
    net.train()
    for k in range(n):
        if args.multi_gpu:
            net.module.blocks[k].eval()
        else:
            net.blocks[k].eval()

    train_loss = 0
    correct = 0
    total = 0
    hinge_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader_classifier):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if args.mse:
            targets_loss = nn.functional.one_hot(targets, num_classes=num_classes).float()
        else:
            targets_loss = targets

        outputs = net.forward([inputs,n])
        loss = criterion_classifier(outputs, targets_loss) + wd_list[n]/2 * torch.norm(net.blocks[n].linear_operator.weight)**2
        #if args.hinge_loss:
        #    curr_hinge_loss = net.hinge_loss([inputs, n], args.squared_hinge)
        #    loss += lambda_hinge_list * curr_hinge_loss
        #    hinge_loss += lambda_hinge_list * curr_hinge_loss.item()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.detach().data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
        
        loss = loss*len(inputs)/len(trainset_class)
        loss.backward()

        progress_bar(batch_idx, len(trainloader_classifier), 'Loss: %.3f | Acc: %.3f%% (%d/%d) |  Hinge Loss: %.3f'
            % (train_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total,hinge_loss/(batch_idx+1)))

    if args.multi_gpu:
        relevant_params = net.module.blocks[n].linear_operator.weight.grad
    else:
        relevant_params = net.blocks[n].linear_operator.weight.grad

    print('Gradient norm', torch.norm(relevant_params))

    optimizer.zero_grad()

def check_dual_qualification(n):
    print('\nChecking qualification constraint for stage: %d' % n)
    net.train()
    for k in range(n):
        if args.multi_gpu:
            net.module.blocks[k].eval()
        else:
            net.blocks[k].eval()

    if args.multi_gpu:
        _ = net.module.blocks[n].get_Z_grad()
    else:
        _ = net.blocks[n].get_Z_grad()
    
    train_loss = 0
    correct = 0
    total = 0
    hinge_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader_classifier):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if args.mse:
            targets_loss = nn.functional.one_hot(targets, num_classes=num_classes).float()
        else:
            targets_loss = targets

        outputs = net.forward([inputs,n], transformed_model=True)
        loss = criterion_classifier(outputs, targets_loss)
        #if args.hinge_loss:
        #    curr_hinge_loss = net.hinge_loss([inputs, n], args.squared_hinge)
        #    loss += lambda_hinge_list * curr_hinge_loss
        #    hinge_loss += lambda_hinge_list * curr_hinge_loss.item()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.detach().data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
        
        loss = loss*len(inputs)/len(trainset_class)
        loss.backward()

        progress_bar(batch_idx, len(trainloader_classifier), 'Loss: %.3f | Acc: %.3f%% (%d/%d) |  Hinge Loss: %.3f'
            % (train_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total,hinge_loss/(batch_idx+1)))

    if args.multi_gpu:
        relevant_params = net.module.blocks[n].get_Z_grad()
        relevant_params = relevant_params.reshape((net.module.blocks[n].P, -1, net.module.blocks[n].in_planes*args.kernel_size*args.kernel_size))
    else:
        relevant_params = net.blocks[n].get_Z_grad()
        relevant_params = relevant_params.reshape((net.blocks[n].P, -1, net.blocks[n].in_planes*args.kernel_size*args.kernel_size))

    gradient_spectral_norm = torch.max(torch.linalg.matrix_norm(relevant_params, ord=2)).item()
    print('Gradient spectral norm bound', gradient_spectral_norm/wd_list[n] - 1)

    optimizer.zero_grad()
    with open(name_log_txt, "a") as text_file:
        print("n: {}, epoch {}, spectral norm {}"
              .format(n,epoch,gradient_spectral_norm/wd_list[n]-1), file=text_file)

n_start = 0

# resume from previously trained checkpoint
if args.resume and args.checkpoint_path != '':
    checkpoint = torch.load(args.checkpoint_path)
    if args.multi_gpu:
        net.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint['model_state_dict'])
    n_start = checkpoint['n']+1
all_outs = [[] for i in range(args.ncnn)]

def test(epoch,n,ensemble=False):
    global acc_test_ensemble
    all_targs = []
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        all_outs[n] = []
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            if args.mse:
                targets_loss = nn.functional.one_hot(targets, num_classes=num_classes).float()
            else:
                targets_loss = targets

            with autocast():
                outputs = net([inputs,n])

                loss = criterion_classifier(outputs, targets_loss)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.detach().data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()
            
            if args.ensemble:
                all_outs[n].append(outputs.data.cpu())
                all_targs.append(targets.data.cpu())

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))

        acc = 100. * float(correct) / float(total)

        if ensemble:
            all_outs[n] = torch.cat(all_outs[n])
            all_targs = torch.cat(all_targs)
            #This is all on cpu so we dont care
            weight = 2 ** (np.arange(n + 1)) / sum(2 ** np.arange(n + 1))
            total_out = torch.zeros((total,10))

            #very lazy
            for i in range(n_start, n+1):
                total_out += float(weight[i])*all_outs[i]


            _, predicted = torch.max(total_out, 1)
            correct = predicted.eq(all_targs).sum()
            acc_ensemble = 100*float(correct)/float(total)
            print('Acc_ensemble: %.2f'%acc_ensemble)
        if ensemble:
            return acc,acc_ensemble
        else:
            return acc

i=0
num_ep = args.nepochs

for n in range(n_start, n_cnn):
    if args.group_norm:
        wd = 0.0
    else:
        wd = wd_list[n]
    print('training stage', n)
    if args.multi_gpu:
        net.module.unfreezeGradient(n)
    else:
        net.unfreezeGradient(n)
    to_train = list(filter(lambda p: p.requires_grad, net.parameters()))
    lr = lr_list[n]
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(to_train, lr=lr, momentum=0.9, weight_decay=wd)
    elif args.optimizer == 'Adam':
        optimizer = optim.AdamW(to_train, lr=lr, weight_decay=wd)

    if not args.reset_momentum:
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.epochdecay, 0.2, verbose=True)

    scaler = GradScaler()

    for epoch in range(0, num_ep):
        print('n: ',n)
        if args.reset_momentum and epoch % args.epochdecay == 0 and epoch > 0:
            lr=lr/5.0
            if args.optimizer == 'SGD':
                optimizer = optim.SGD(to_train, lr=lr, momentum=0.9, weight_decay=wd)
            elif args.optimizer == 'Adam':
                optimizer = optim.AdamW(to_train, lr=lr, weight_decay=wd)
            print('new lr:',lr)
        acc_train, loss_train = train_classifier(epoch,n)
        if args.ensemble:
            acc_test,acc_test_ensemble = test(epoch,n,args.ensemble)

            with open(name_log_txt, "a") as text_file:
                print("n: {}, epoch {}, train loss {}, train {}, test {},ense {} "
                      .format(n,epoch,loss_train,acc_train,acc_test,acc_test_ensemble), file=text_file)
        else:
            acc_test = test(epoch, n)
            with open(name_log_txt, "a") as text_file:
                print("n: {}, epoch {}, train loss {}, train {}, test {}, ".format(n,epoch,loss_train,acc_train,acc_test), file=text_file)

        if args.debug:
            break
        if not args.reset_momentum:
            scheduler.step()

    if args.burer_monteiro and args.check_stationary:
        check_stationarity(n)

    if args.burer_monteiro and args.check_constraint:
        check_dual_qualification(n)
    
    del to_train
    del optimizer
    if not args.reset_momentum:
        del scheduler
    gc.collect()
    torch.cuda.empty_cache()

    if args.save_checkpoint:
        curr_sv_model = name_save_model + '_' + str(n) + '.pt'
        print('saving checkpoint')
        if args.multi_gpu:
            torch.save({
                    'n': n,
                    'model_state_dict': net.module.state_dict(),
                    }, curr_sv_model)
        else:
            torch.save({
                    'n': n,
                    'model_state_dict': net.state_dict(),
                    }, curr_sv_model)


if args.e2e_epochs > 0:
    if args.multi_gpu:
        net.module.unfreezeAll()
    else:
        net.unfreezeAll()
    to_train = list(filter(lambda p: p.requires_grad, net.parameters()))

    lr = lr_list[-1]
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(to_train, lr=lr, momentum=0.9, weight_decay=wd)
    elif args.optimizer == 'Adam':
        optimizer = optim.AdamW(to_train, lr=lr, weight_decay=wd)

    if not args.reset_momentum:
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.epochdecay, 0.2, verbose=True)

    for epoch in range(0, args.e2e_epochs):
        if args.reset_momentum and epoch % args.epochdecay == 0 and epoch > 0:
            lr=lr/5.0
            if args.optimizer == 'SGD':
                optimizer = optim.SGD(to_train, lr=lr, momentum=0.9, weight_decay=wd)
            elif args.optimizer == 'Adam':
                optimizer = optim.AdamW(to_train, lr=lr, weight_decay=wd)
            print('new lr:',lr)


        acc_train = train_classifier(epoch,n_cnn-1)
        if args.ensemble:
            acc_test,acc_test_ensemble = test(epoch,n_cnn-1,args.ensemble)

            with open(name_log_txt, "a") as text_file:
                print("e2e: epoch {}, train {}, test {},ense {} "
                      .format(epoch,acc_train,acc_test,acc_test_ensemble), file=text_file)
        else:
            acc_test = test(epoch, n)
            with open(name_log_txt, "a") as text_file:
                print('e2e: epoch {}, train {}, test {}, '.format(epoch,acc_train,acc_test), file=text_file)

        if args.debug:
            break

        if not args.reset_momentum:
            scheduler.step()
    
    del to_train
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()

    if args.save_checkpoint:
        curr_sv_model = name_save_model + '_e2e.pt'
        print('saving checkpoint')
        if args.multi_gpu:
            torch.save({
                    'n': n_cnn-1,
                    'model_state_dict': net.module.state_dict(),
                    }, curr_sv_model)
        else:
            torch.save({
                    'n': n_cnn-1,
                    'model_state_dict': net.state_dict(),
                    }, curr_sv_model)

