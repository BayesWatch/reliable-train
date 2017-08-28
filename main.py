'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from torch.autograd import Variable

import learn

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--schedule_period', default=30, type=float, help='learning rate schedule period')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--data', '-d', default=os.environ.get('SCRATCH',os.getcwd()), help='place to store data')
parser.add_argument('--sgdr', action='store_true', help='use the SGDR learning rate schedule')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

# count how many gpus we're working with here
n_gpus = torch.cuda.device_count()

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

class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds)>=offset+length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        return self.parent_ds[i+self.offset]

data_save_loc = os.path.join(args.data, 'data')
print("Saving data and checkpoints to: %s"%data_save_loc)
trainvalset = torchvision.datasets.CIFAR10(root=data_save_loc, train=True, download=True, transform=transform_train)
trainset = PartialDataset(trainvalset, 0, 40000)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

valset = PartialDataset(trainvalset, 40000, 10000)
valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_save_loc, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

checkpoint_loc = os.path.join(data_save_loc, 'checkpoint')

# Make an initial checkpoint if we don't have one
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_loc)
    checkpoint = torch.load(os.path.join(checkpoint_loc, 'ckpt.t7'))
else:
    print('==> Building model..')
    net = VGG('VGG16')
    if use_cuda:
        net.cuda()
    # net = ResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    checkpoint = learn.save_checkpoint(checkpoint_loc, net, 0.0, 0)

checkpoints = []
for i in range(2):
    net = VGG('VGG16')
    checkpoint = learn.save_checkpoint(checkpoint_loc, net, 0.0, 0)
    checkpoints.append(checkpoint)

period = args.schedule_period*len(trainloader)
if not args.sgdr:
    print("Using standard two step learning rate schedule with period %i minibatches."%period)
    lr_schedule = lambda batch_idx: learn.standard_schedule(period, batch_idx)
else:
    print("Using SGDR with period %i minibatches."%period)
    lr_schedule = lambda batch_idx: learn.sgdr(period, batch_idx)

for epoch in range(200):
    learn.train(epoch, checkpoints, trainloader, args.lr, lr_schedule)
    #assert False
    #best_acc = learn.validate(epoch, checkpoints, valloader, checkpoint_loc)
