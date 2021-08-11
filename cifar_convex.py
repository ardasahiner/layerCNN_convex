"Greedy layerwise cifar training"
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

from model_greedy import *
from model_greedy_convex import *
from torch.autograd import Variable

from utils import *

from random import randint
import random
import datetime
import json
import dask.array as da

# prepare dataset
class PrepareData(torch.utils.data.Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PrepareData3D(torch.utils.data.Dataset):
    def __init__(self, X, y, z):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X

        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

        if not torch.is_tensor(z):
            self.z = torch.from_numpy(z)
        else:
            self.z = z


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.z[idx]

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ncnn',  default=1,type=int, help='depth of the CNN')
parser.add_argument('--nepochs',  default=50,type=int, help='number of epochs')
parser.add_argument('--epochdecay',  default=15,type=int, help='number of epochs')
parser.add_argument('--avg_size',  default=3,type=int, help='size of averaging ')
parser.add_argument('--feature_size',  default=256,type=int, help='feature size')
parser.add_argument('--ds-type', default=None, help="type of downsampling. Defaults to old block_conv with psi. Options 'psi', 'stride', 'avgpool', 'maxpool'")
parser.add_argument('--ensemble', default=1,type=int,help='compute ensemble')
parser.add_argument('--name', default='',type=str,help='name')
parser.add_argument('--batch_size', default=250,type=int,help='batch size')
parser.add_argument('--bn', default=1,type=int,help='use batchnorm')
parser.add_argument('--debug', default=0,type=int,help='debug')
parser.add_argument('--debug_parameters', default=0,type=int,help='verification that layers frozen')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 6)')
parser.add_argument('--down', default='[2, 3]', type=str,
                        help='layer at which to downsample')
parser.add_argument('--sparsity', default=0.1, type=float,
                        help='sparsity of hyperplane generating arrangements')
parser.add_argument('--feat_agg', default='weight_rankone', type=str,
                        help='way to aggregate features from layer to layer')
parser.add_argument('--multi_gpu', default=0, type=int,
                        help='use multiple gpus')
parser.add_argument('--seed', default=None, help="Fixes the CPU and GPU random seeds to a specified number")
parser.add_argument('--save_dir', '-sd', default='checkpoints/', help='directory to save checkpoints into')
parser.add_argument('--checkpoint_path', '-cp', default='', help='path to checkpoint to load')
parser.add_argument('--deterministic', '-det', action='store_true', help='Deterministic operations for numerical stability')
parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save checkpoints')
parser.add_argument('--optimizer', default='Adam', help='What optimizer to use')
parser.add_argument('--data_dir', default='/mnt/dense/sahiner', help='Dataset directory')
parser.add_argument('--group_norm', action='store_true', help='Whether to use group norm penalty (otherwise, standard weight decay is used)')
parser.add_argument('--wd', default=5e-4, type=float, help='regularization parameter')
parser.add_argument('--mse', action='store_true', help='Whether to use MSE loss (otherwise, softmax cross-entropy is used)')
parser.add_argument('--nonconvex_stages', default='[]', type=str, help='layers at which to use nonconvex block')
parser.add_argument('--trunc_idx', default=20, type=int, help='How much to truncate SVD while using BN')

args = parser.parse_args()
opts = vars(args)
args.ensemble = args.ensemble>0
args.bn = args.bn > 0
if args.sparsity == 0:
    args.sparsity = None

args.debug_parameters = args.debug_parameters > 0
args.multi_gpu = args.multi_gpu > 0

if args.debug:
    args.nepochs = 1 # we run just one epoch per greedy layer training in debug mode

downsample =  list(np.array(json.loads(args.down)))
args.nonconvex_stages =  list(np.array(json.loads(args.nonconvex_stages)))
in_size=32
mode=0

if args.seed is not None:
    seed = int(args.seed)
    random.seed(a=args.seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

save_name = 'convex_layersize_'+str(args.feature_size) +'down_' +  args.down 
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

s = None
v = None

if args.bn:
    print('==> Preparing data..')
    train_dataset = torchvision.datasets.CIFAR10(
        args.data_dir, train=True, download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))

    test_dataset = torchvision.datasets.CIFAR10(
        args.data_dir, train=False, download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))

    dummy_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=50000, shuffle=False,
            pin_memory=True, sampler=None)

    print('loading train data')
    for A, y in dummy_loader:
        break

    dummy_test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=10000, shuffle=False,
            pin_memory=True, sampler=None)

    print('loading test data')
    for A_test, y_test in dummy_test_loader:
        break

    A_orig = A
    A_test_orig = A_test

    kernel_size = 3
    padding =2
    stride = 3
    unf = nn.Unfold(kernel_size=kernel_size, padding=padding,stride=stride)
    A_conv = unf(A).permute(0, 2, 1)
    A_test_conv = unf(A_test).permute(0, 2, 1)

    n, k, h= A_conv.shape
    n_test = A_test_conv.shape[0]

    M = A_conv.reshape((-1, kernel_size*kernel_size*3))
    M_test = A_test_conv.reshape((-1, M.shape[-1]))
    
    train_means = torch.mean(M, dim=0)
    M = M - train_means
    M_test = M_test - train_means
    
    M_da = da.from_array(M.numpy())
    u, s, v = da.linalg.svd(M_da)

    s_orig = s
    # whiten the dataset
    print('whitening...')
    if args.trunc_idx > 0:
        u = u[:, :args.trunc_idx]
        s = s[:args.trunc_idx]
        v = v[:args.trunc_idx, :]

    start = time.time()
    print('computing svd')
    u = torch.from_numpy(u.compute())
    s = torch.from_numpy(s.compute())
    v = torch.from_numpy(v.compute())
    print('computing svd took', time.time()-start)

    print(s)
    s_orig = s_orig.compute()
    var_explained = np.cumsum(s_orig**2)/np.sum(s_orig**2)
    print(var_explained)
    if args.trunc_idx > 0:
        print(var_explained[args.trunc_idx])

    print('forming new dataset')
    A_train = (u).reshape((n, k, -1))
    A_test = M_test @ v.t()
    A_test = A_test @ torch.diag(1/s)
    A_test = A_test.reshape((n_test, k, -1))

    if A_train.shape[-1] < h:
        A_train = torch.cat((A_train, torch.zeros((n, k, h-A_train.shape[-1]))), dim=2)
        A_test = torch.cat((A_test, torch.zeros((n_test, k, h-A_test.shape[-1]))), dim=2)

    fold = nn.Fold(output_size=32, padding=padding, kernel_size=kernel_size, stride=stride)
    A_train = fold(A_train.permute(0, 2,1))
    A_test = fold(A_test.permute(0, 2, 1))
    
    trainset_class = PrepareData3D(A_train, y, A_orig)
    trainloader_classifier = torch.utils.data.DataLoader(trainset_class, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testset = PrepareData3D(A_test, y_test, A_test_orig)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

else:
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

    trainset_class = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True,transform=transform_train)
    trainloader_classifier = torch.utils.data.DataLoader(trainset_class, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Model

print('==> Building model..')
n_cnn=args.ncnn
net = convexGreedyNet(custom_cvx_layer, n_cnn, args.feature_size, avg_size=args.avg_size,
                      downsample=downsample, batchnorm=args.bn, sparsity=args.sparsity, feat_aggregate=args.feat_agg,
                      nonconvex_block=block_conv, nonconvex_stages=args.nonconvex_stages, s=s, v=v)
    
with open(name_log_txt, "a") as text_file:
    print(net, file=text_file)


if args.multi_gpu:
    net = torch.nn.DataParallel(net).cuda()
net = net.cuda()
cudnn.benchmark = True
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
    for batch_idx, (inputs, targets, orig_inputs) in enumerate(trainloader_classifier):
        if use_cuda:
            inputs, targets,orig_inputs = inputs.cuda(), targets.cuda(), orig_inputs.cuda()

        if args.mse:
            targets_loss = nn.functional.one_hot(targets, num_classes=10).float()
        else:
            targets_loss = targets

        optimizer.zero_grad()
        outputs = net.forward([inputs, orig_inputs, n])

        loss = criterion_classifier(outputs, targets_loss)

        if args.group_norm:
            if args.multi_gpu:
                loss += args.wd/2*torch.sum(torch.norm(net.module.blocks[n].v.reshape((net.module.blocks[n].v.shape[0], -1)), dim=1))
            else:
                loss += args.wd/2*torch.sum(torch.norm(net.blocks[n].v.reshape((net.blocks[n].v.shape[0], -1)), dim=1))

        loss.backward()
        optimizer.step()
        loss_pers=0

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

        train_loss += loss.item()
        _, predicted = torch.max(outputs.detach().data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(trainloader_classifier), 'Loss: %.3f | Acc: %.3f%% (%d/%d) |  losspers: %.3f'
            % (train_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total,loss_pers))


    acc = 100.*float(correct)/float(total)
    return acc

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
        for batch_idx, (inputs, targets, orig_inputs) in enumerate(testloader):
            if use_cuda:
                inputs, targets, orig_inputs = inputs.cuda(), targets.cuda(), orig_inputs.cuda()

            if args.mse:
                targets_loss = nn.functional.one_hot(targets, num_classes=10).float()
            else:
                targets_loss = targets

            outputs = net([inputs, orig_inputs, n])

            loss = criterion_classifier(outputs, targets_loss)

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
if args.group_norm:
    wd = 0.0
else:
    wd = args.wd

for n in range(n_start, n_cnn):
    print('training stage', n)
    if n == 2:
        cudnn.benchmark = False
    if args.multi_gpu:
        net.module.unfreezeGradient(n)
    else:
        net.unfreezeGradient(n)
    to_train = list(filter(lambda p: p.requires_grad, net.parameters()))

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(to_train, lr=args.lr, momentum=0.9, weight_decay=wd)
    elif args.optimizer == 'Adam':
        optimizer = optim.AdamW(to_train, lr=args.lr, weight_decay=wd)

    num_decays = num_ep // args.epochdecay
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               [(i+1)*args.epochdecay for i in range(num_decays)],
                                               0.5, verbose=True)

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

    # decay the learning rate at each stage
    if n == 1:
        args.lr /= 100

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

