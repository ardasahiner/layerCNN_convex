"Greedy layerwise cifar training"
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import gc

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from model_greedy_convex import *
from torch.autograd import Variable

from utils import *

from random import randint
import datetime
import json



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ncnn',  default=5,type=int, help='depth of the CNN')
parser.add_argument('--nepochs',  default=50,type=int, help='number of epochs')
parser.add_argument('--epochdecay',  default=20,type=int, help='number of epochs')
parser.add_argument('--avg_size',  default=16,type=int, help='size of averaging ')
parser.add_argument('--feature_size',  default=64,type=int, help='feature size')
parser.add_argument('--ds-type', default=None, help="type of downsampling. Defaults to old block_conv with psi. Options 'psi', 'stride', 'avgpool', 'maxpool'")
parser.add_argument('--nlin',  default=0,type=int, help='nlin')
parser.add_argument('--ensemble', default=1,type=int,help='compute ensemble')
parser.add_argument('--name', default='',type=str,help='name')
parser.add_argument('--batch_size', default=5,type=int,help='batch size')
parser.add_argument('--bn', default=0,type=int,help='use batchnorm')
parser.add_argument('--debug', default=0,type=int,help='debug')
parser.add_argument('--debug_parameters', default=0,type=int,help='verification that layers frozen')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 6)')
parser.add_argument('--width_aux', default=128,type=int,help='auxillary width')
parser.add_argument('--down', default='[2, 3]', type=str,
                        help='layer at which to downsample')
parser.add_argument('--sparsity', default=0.0, type=float,
                        help='sparsity of hyperplane generating arrangements')
parser.add_argument('--feat_agg', default='random', type=str,
                        help='way to aggregate features from layer to layer')
parser.add_argument('--seed', default=None, help="Fixes the CPU and GPU random seeds to a specified number")

args = parser.parse_args()
opts = vars(args)
args.ensemble = args.ensemble>0
args.bn = args.bn > 0
if args.sparsity == 0:
    args.sparsity = None

assert args.bn == False, 'batch norm not yet implemented'
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

save_name = 'convex_layersize_'+str(args.feature_size)+'width_' \
            + str(args.width_aux) + 'depth_' + str(args.nlin) + 'ds_type_' + str(args.ds_type) +'down_' +  args.down 
#logging
time_stamp = str(datetime.datetime.now().isoformat())

name_log_dir = ''.join('{}{}-'.format(key, val) for key, val in sorted(opts.items()))+time_stamp
name_log_dir = 'runs/'+name_log_dir

name_log_txt = time_stamp + save_name + str(randint(0, 1000)) + args.name
debug_log_txt = name_log_txt + '_debug.log'
name_save_model = name_log_txt + '.t7'
name_log_txt=name_log_txt   +'.log'

with open(name_log_txt, "a") as text_file:
    print(opts, file=text_file)

use_cuda = torch.cuda.is_available()
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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

trainset_class = torchvision.datasets.CIFAR10(root='/mnt/raid3/sahiner/cifar10', train=True, download=True,transform=transform_train)
trainloader_classifier = torch.utils.data.DataLoader(trainset_class, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
testset = torchvision.datasets.CIFAR10(root='/mnt/raid3/sahiner/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Model

print('==> Building model..')
n_cnn=args.ncnn
net = convexGreedyNet(custom_cvx_layer, n_cnn, args.feature_size, avg_size=args.avg_size,
                      downsample=downsample, batchnorm=args.bn, sparsity=args.sparsity, feat_aggregate=args.feat_agg)
    
with open(name_log_txt, "a") as text_file:
    print(net, file=text_file)

net = net.cuda()
cudnn.benchmark = True

criterion_classifier = nn.CrossEntropyLoss()

criterion = nn.CrossEntropyLoss()

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

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net.forward([inputs,n])

        # TODO: add appropriate group norm regularizer
        loss = criterion_classifier(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_pers=0

        if args.debug_parameters:

            s_dict = net.state_dict()
            with open(debug_log_txt, "a") as text_file:
                print("iteration %d" % (batch_idx), file=text_file)
                for param in s_dict.keys():
                    diff = np.sum((s_dict[param].cpu().numpy()-net_cpu_dict[param])**2)
                    print("n: %d parameter: %s size: %s changed by %.5f" % (n,param,net_cpu_dict[param].shape,diff),file=text_file)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.detach().data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(trainloader_classifier), 'Loss: %.3f | Acc: %.3f%% (%d/%d) |  losspers: %.3f'
            % (train_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total,loss_pers))

    acc = 100.*float(correct)/float(total)
    return acc

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

            outputs = net([inputs,n])

            loss = criterion_classifier(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.detach().data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))

            if args.ensemble:
                all_outs[n].append(outputs.data.cpu())
                all_targs.append(targets.data.cpu())
        acc = 100. * float(correct) / float(total)

        if ensemble:
            all_outs[n] = torch.cat(all_outs[n])
            all_targs = torch.cat(all_targs)
            #This is all on cpu so we dont care
            weight = 2 ** (np.arange(n + 1)) / sum(2 ** np.arange(n + 1))
            total_out = torch.zeros((total,10))

            #very lazy
            for i in range(n+1):
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

for n in range(n_cnn):
    net.unfreezeGradient(n)
    to_train = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optim.SGD(to_train, lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.epochdecay, 0.2, verbose=True)

    for epoch in range(0, num_ep):
        acc_train = train_classifier(epoch,n)
        if args.ensemble:
            acc_test,acc_test_ensemble = test(epoch,n,args.ensemble)

            with open(name_log_txt, "a") as text_file:
                print("n: {}, epoch {}, train {}, test {},ense {} "
                      .format(n,epoch,acc_train,acc_test,acc_test_ensemble), file=text_file)
        else:
            acc_test = test(epoch, n)
            with open(name_log_txt, "a") as text_file:
                print("n: {}, epoch {}, train {}, test {}, ".format(n,epoch,acc_train,acc_test), file=text_file)

        if args.debug:
            break

        scheduler.step()
    
    del to_train
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache() 

state_final = {
            'net': net,
            'acc_test': acc_test,
            'acc_train': acc_train,
        }
torch.save(state_final,save_name)
