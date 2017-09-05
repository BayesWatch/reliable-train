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
from utils import gridfile_parse, existing_checkpoints, write_status

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate (only used if resuming)')
parser.add_argument('--schedule_period', default=30, type=float, help='learning rate schedule period (only used if resuming)')
parser.add_argument('--resume', '-r', default=False, type=str, help='checkpoint to resume')
parser.add_argument('--data', '-d', default=os.environ.get('SCRATCH',os.getcwd()), help='place to store data')
parser.add_argument('--sgdr', action='store_true', help='use the SGDR learning rate schedule')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

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

model = 'VGG16'
checkpoint_loc = os.path.join(data_save_loc, 'checkpoint', model)

def init_model(lr, period):
    net = VGG(model)
    checkpoint = learn.save_checkpoint(checkpoint_loc, net, 0.0, 0, float(lr), int(period))
    checkpoint['period'] = int(period)
    checkpoint['init_lr'] = float(lr)
    return checkpoint

def load_model(checkpoint_abspath, lr, period):
    checkpoint = torch.load(checkpoint_abspath)
    checkpoint['period'] = int(period)
    checkpoint['init_lr'] = float(lr)
    return checkpoint

# Make an initial checkpoint if we don't have one
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_loc)
    checkpoints = [torch.load(os.path.join(checkpoint_loc, args.resume))]
    period = args.schedule_period*len(trainloader)
else:
    # enumerate grid search settings
    with open("grid_%s.csv"%("sgdr" if args.sgdr else "default"), "r") as f:
        grid_settings = gridfile_parse(f)
    from collections import OrderedDict
    checkpoint_loaders = OrderedDict()
    # load any we have checkpoints for
    existing = existing_checkpoints(checkpoint_loc)
    # initialise those we don't
    for setting in grid_settings:
        if setting in existing.keys():
            abspath = existing[setting]['abspath']
            checkpoint_loaders[setting] = lambda s: load_model(abspath, s[0], s[1])
        else:
            checkpoint_loaders[setting] = lambda s: init_model(*s)

if not args.sgdr:
    print("Using standard two step learning rate schedule.")
    lr_schedule = lambda period, batch_idx: learn.standard_schedule(period, batch_idx)
else:
    print("Using SGDR.")
    lr_schedule = lambda period, batch_idx: learn.sgdr(period, batch_idx)

while True:
    settings_to_run = [grid_settings.pop(0) for i in range(n_gpus)] 
    grid_settings += settings_to_run # put these back on the end
    checkpoints = [checkpoint_loaders[s](s) for s in settings_to_run]
    # get the next
    learn.train(checkpoints, trainloader, lr_schedule)
    learn.validate(checkpoints, valloader, checkpoint_loc, save=True)
    # update checkpoint_loaders and delete checkpoints
    for i, setting in enumerate(settings_to_run):
        if 'recent_abspath' in checkpoints[i].keys():
            new_abspath = checkpoints[i]['recent_abspath']
            checkpoint_loaders[setting] = lambda s: load_model(new_abspath, s[0], s[1])
    del checkpoints
    # write results to log file
    write_status('grid.log', checkpoint_loc, args.sgdr)
