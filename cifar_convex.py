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

from model_greedy_convex import *
from torch.autograd import Variable

from utils import *

from random import randint
import datetime
import json

from typing import List


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', nargs='+', default=[0.1, 0.01, 0.001, 0.01, 0.01], type=float, help='learning rate')
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
parser.add_argument('--debug', default=0,type=int,help='debug')
parser.add_argument('--debug_parameters', default=0,type=int,help='verification that layers frozen')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 6)')
parser.add_argument('--down', default='[2, 3]', type=str,
                        help='layer at which to downsample')

parser.add_argument('--gpu', default=None, type=int, help='Which GPU to use')

parser.add_argument('--seed', default=0, help="Fixes the CPU and GPU random seeds to a specified number")
parser.add_argument('--save_dir', '-sd', default='checkpoints/', help='directory to save checkpoints into')
parser.add_argument('--checkpoint_path', '-cp', default='', help='path to checkpoint to load')
parser.add_argument('--deterministic', '-det', action='store_true', help='Deterministic operations for numerical stability')
parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save checkpoints')
parser.add_argument('--optimizer', default='SGD', help='What optimizer to use')

parser.add_argument('--data_dir', default='', help='Dataset directory')

parser.add_argument('--wd', nargs='+', default=[5e-4], type=float, help='regularization parameter')
parser.add_argument('--mse', action='store_true', help='Whether to use MSE loss (otherwise, softmax cross-entropy is used)')

parser.add_argument('--kernel_size', default=3, type=int, help='kernel size of convolutions')

parser.add_argument('--burer_monteiro', action='store_true', help='Whether to use burer-monteiro factorization')
parser.add_argument('--burer_dim', default=1, type=int, help='dimension of burer monteiro')

parser.add_argument('--data_set', default='CIFAR10', choices=['CIFAR10', 'FMNIST'],
                    type=str, help='Dataset name')

args = parser.parse_args()
opts = vars(args)

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
args.ensemble = args.ensemble>0

if len(args.lr) == 1:
    lr_list = [args.lr[0]]*args.ncnn
else:
    lr_list = args.lr
if len(args.wd) == 1:
    wd_list = [args.wd[0]]*args.ncnn
else:
    wd_list = args.wd

for i in range(args.ncnn):
    if wd_list[i] < 0:
        wd_list[i] = 10**wd_list[i]
    if lr_list[i] < 0:
        lr_list[i] = 10**lr_list[i]

assert args.kernel_size %2 == 1, 'kernel size must be odd'
args.debug_parameters = args.debug_parameters > 0

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

if args.data_set == 'CIFAR10' or args.data_set == 'FMNIST':
    num_classes = 10
else:
    assert False, "dataset name not in CIFAR10, FMNIST"

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
# Model

print('==> Building model..')
n_cnn=args.ncnn

net = convexGreedyNet(custom_cvx_layer, n_cnn, args.feature_size, in_size=in_size, avg_size=args.avg_size, num_classes=num_classes,
                      downsample=downsample, kernel_size=args.kernel_size, burer_monteiro=args.burer_monteiro, burer_dim=args.burer_dim, 
                      in_planes = in_planes)
    
with open(name_log_txt, "a") as text_file:
    print(net, file=text_file)

n_parameters = sum(p.numel() for p in net.parameters())
print('number of params:', n_parameters)

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
        net.blocks[k].eval()
    
    if args.debug_parameters:
    #This is used to verify that early layers arent updated
        import copy
        #store all parameters on cpu as numpy array
        net_cpu = copy.deepcopy(net).cpu()
        net_cpu_dict = net_cpu.state_dict()
        with open(debug_log_txt, "a") as text_file:
            print('n: %d'%n)
            for param in net_cpu_dict.keys():
                net_cpu_dict[param]=net_cpu_dict[param].numpy()
                print("parameter stored on cpu as numpy: %s  "%(param),file=text_file)

    train_loss = 0
    correct = 0
    total = 0
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
            s_dict = net.state_dict()
            with open(debug_log_txt, "a") as text_file:
                print("iteration %d" % (batch_idx), file=text_file)
                for param in s_dict.keys():
                    diff = np.sum((s_dict[param].cpu().numpy()-net_cpu_dict[param])**2)
                    print("n: %d parameter: %s size: %s changed by %.5f" % (n,param,net_cpu_dict[param].shape,diff),file=text_file)


        progress_bar(batch_idx, len(trainloader_classifier), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))

    acc = 100.*float(correct)/float(total)
    return acc, train_loss/(batch_idx+1)

n_start = 0

# resume from previously trained checkpoint
if args.resume and args.checkpoint_path != '':
    checkpoint = torch.load(args.checkpoint_path)
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
    wd = wd_list[n]
    print('training stage', n)
    net.unfreezeGradient(n)
    to_train = list(filter(lambda p: p.requires_grad, net.parameters()))
    lr = lr_list[n]
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(to_train, lr=lr, momentum=0.9, weight_decay=wd)
    elif args.optimizer == 'Adam':
        optimizer = optim.AdamW(to_train, lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.epochdecay, 0.2, verbose=True)

    scaler = GradScaler()

    for epoch in range(0, num_ep):
        print('n: ',n)
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
        scheduler.step()

    
    del to_train
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()

    if args.save_checkpoint:
        curr_sv_model = name_save_model + '_' + str(n) + '.pt'
        print('saving checkpoint')
        torch.save({
                'n': n,
                'model_state_dict': net.state_dict(),
                }, curr_sv_model)

