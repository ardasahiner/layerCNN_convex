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
parser.add_argument('--epochdecay',  nargs='+', default=[15],type=int, help='number of epochs')
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


parser.add_argument('--decompose', action='store_true', help='Whether to decompose Gated ReLU weights into ReLU')
parser.add_argument('--lambd', default=1e-10, type=float, help='Lambda for cone decomposition')
parser.add_argument('--decomp_epochs', default=5, type=int, help='Over how many epochs to compute cone decomposition')
parser.add_argument('--decomp_lr', nargs='+', default=[0.1], type=float, help='LR to use for cone decomposition')

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
if len(args.epochdecay) == 1:
    decay_list = [args.epochdecay[0]]*args.ncnn
else:
    decay_list = args.epochdecay
if len(args.wd) == 1:
    wd_list = [args.wd[0]]*args.ncnn
else:
    wd_list = args.wd
if len(args.lambda_hinge_loss) == 1:
    lambda_hinge_list = [args.lambda_hinge_loss[0]]*args.ncnn
else:
    lambda_hinge_list = args.lambda_hinge_loss
if len(args.decomp_lr) == 1:
    decomp_lr_list = [args.decomp_lr[0]]*args.ncnn
else:
    decomp_lr_list = args.decomp_lr

ncnn = args.ncnn
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
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    mean = np.expand_dims(mean, (0, 2, 3)) # 1 x 3 x 1 x 1
    std = np.expand_dims(std, (0, 2, 3)) # 1 x 3 x 1 x 1

    X = (X/255 - mean)/std

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
                      sign_pattern_bias=sign_pattern_bias, relu=args.relu, in_planes = in_planes, decompose=args.decompose, lambd=args.lambd)
    
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

def train_classifier(epoch):
    print('\nSubepoch: %d' % epoch)
    net.train()
    
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

    train_loss = [0]*ncnn
    correct = [0]*ncnn
    total = [0]*ncnn
    for batch_idx, (inputs, targets) in enumerate(trainloader_classifier):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if args.mse:
            targets_loss = nn.functional.one_hot(targets, num_classes=num_classes).float()
        else:
            targets_loss = targets

        for n in range(ncnn):
            # Forward
            layer_optim[n].zero_grad()
            
            with autocast():
                outputs = net([inputs, n], store_activations=True)
                targets_loss = targets_loss.to(outputs.device)
                loss = criterion_classifier(outputs, targets_loss)

            scaler.scale(loss).backward()
            scaler.step(layer_optim[n])
            scaler.update()

            # measure accuracy and record loss
            train_loss[n] += loss.item()

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)
            
            _, predicted = torch.max(outputs.detach().data, 1)
            total[n] += targets.size(0)
            correct[n] += predicted.eq(targets.data).cpu().sum().item()
            
        torch.cuda.synchronize()
        
        progress_bar(batch_idx, len(trainloader_classifier), "Loss: {} | Acc: {}"
                .format(np.array(train_loss)/(batch_idx+1), 100.*np.array(correct)/np.array(total)))

    acc = 100.*float(correct[-1])/float(total[-1])
    return acc, train_loss[-1]/(batch_idx+1)


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

def test(epoch,ensemble=False):
    global acc_test_ensemble
    all_targs = []
    net.eval()
    test_loss = [0]*ncnn
    correct = [0]*ncnn
    total = [0]*ncnn
    
    all_outs = [[] for i in range(args.ncnn)]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            if args.mse:
                targets_loss = nn.functional.one_hot(targets, num_classes=num_classes).float()
            else:
                targets_loss = targets

            for n in range(ncnn):
                # Forward
                layer_optim[n].zero_grad()
                
                with autocast():
                    outputs = net([inputs, n], store_activations=True)
                    targets_loss = targets_loss.to(outputs.device)
                    loss = criterion_classifier(outputs, targets_loss)
            
                if ensemble:
                    all_outs[n].append(outputs.data.cpu())
                    if n == 0:
                        all_targs.append(targets.data.cpu())

                # measure accuracy and record loss
                test_loss[n] += loss.item()

                if not math.isfinite(loss.item()):
                    print("Loss is {}, stopping training".format(loss.item()))
                    sys.exit(1)
                
                _, predicted = torch.max(outputs.detach().data, 1)
                total[n] += targets.size(0)
                correct[n] += predicted.eq(targets.data).cpu().sum().item()
                
            torch.cuda.synchronize()
        
            progress_bar(batch_idx, len(testloader), "Test Loss: {} | Acc: {}"
                    .format(np.array(test_loss)/(batch_idx+1), 100.*np.array(correct)/np.array(total)))

        acc = 100. * float(correct[-1]) / float(total[-1])
        
        if ensemble:
            all_targs = torch.cat(all_targs)
        for n in range(ncnn):
            if ensemble:
                all_outs[n] = torch.cat(all_outs[n])
                #This is all on cpu so we dont care
                weight = 2 ** (np.arange(n + 1)) / sum(2 ** np.arange(n + 1))
                total_out = torch.zeros((total[-1],10))

                #very lazy
                for i in range(n_start, n+1):
                    total_out += float(weight[i])*all_outs[i]

                _, predicted = torch.max(total_out, 1)
                correct = predicted.eq(all_targs).sum()
                acc_ensemble = 100*float(correct)/float(total[-1])
                print('Acc_ensemble: %.2f'%acc_ensemble)

        if ensemble:
            return acc,acc_ensemble
        else:
            return acc

num_ep = args.nepochs
layer_optim = [None]*ncnn
layer_scheduler = [None]*ncnn

for n in range(n_start, n_cnn):
    if args.multi_gpu:
        to_train = list(filter(lambda p: p.requires_grad, net.module.blocks[n].parameters()))
    else:
        to_train = list(filter(lambda p: p.requires_grad, net.blocks[n].parameters()))
    lr = lr_list[n]
    if args.optimizer == 'SGD':
        layer_optim[n] = optim.SGD(to_train, lr=lr, momentum=0.9, weight_decay=wd_list[n])
    elif args.optimizer == 'Adam':
        layer_optim[n] = optim.AdamW(to_train, lr=lr, weight_decay=wd_list[n])

    layer_scheduler[n] = optim.lr_scheduler.StepLR(layer_optim[n], decay_list[n], 0.2, verbose=True)

scaler = GradScaler()

for epoch in range(0, num_ep):
    acc_train, loss_train = train_classifier(epoch)
    if args.ensemble:
        acc_test,acc_test_ensemble = test(epoch,args.ensemble)

        with open(name_log_txt, "a") as text_file:
            print("epoch {}, train loss {}, train {}, test {},ense {} "
                  .format(epoch,loss_train,acc_train,acc_test,acc_test_ensemble), file=text_file)
    else:
        acc_test = test(epoch)
        with open(name_log_txt, "a") as text_file:
            print("epoch {}, train loss {}, train {}, test {}, ".format(epoch,loss_train,acc_train,acc_test), file=text_file)

    if args.debug:
        break

    for n in range(ncnn):
        layer_scheduler[n].step()

    if args.save_checkpoint:
        curr_sv_model = name_save_model + '_' + str(n) + '.pt'
        print('saving checkpoint')
        if args.multi_gpu:
            torch.save({
                    'model_state_dict': net.module.state_dict(),
                    }, curr_sv_model)
        else:
            torch.save({
                    'model_state_dict': net.state_dict(),
                    }, curr_sv_model)


