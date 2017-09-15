'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import sys
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

from models import *
from torch.autograd import Variable

import numpy as np

from utils import gridfile_parse, existing_checkpoints, write_status, \
        clean_checkpoints, get_summary_writer, progress_bar
from checkpoint import Checkpoint
from hyperband import Hyperband

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data', '-d', default=os.environ.get('SCRATCH',os.getcwd()), help='place to store data')
#parser.add_argument('--sgdr', action='store_true', help='use the SGDR learning rate schedule')
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
print("Data saved to: %s"%data_save_loc)
trainvalset = torchvision.datasets.CIFAR10(root=data_save_loc, train=True, download=True, transform=transform_train)
trainset = PartialDataset(trainvalset, 0, 40000)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

valset = PartialDataset(trainvalset, 40000, 10000)
valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_save_loc, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model_tag = 'VGG16'
checkpoint_loc = os.path.join(args.data, 'checkpoint', model_tag)
if not os.path.isdir(checkpoint_loc):
    os.makedirs(checkpoint_loc)
print("Checkpoints saved to: %s"%checkpoint_loc)

log_loc = os.path.join(args.data, 'logs', model_tag)
if not os.path.isdir(log_loc):
    os.makedirs(log_loc)
print("Tensorboard logs saved to: %s"%log_loc)

def standard_schedule(decay_ratio, period, batch_idx):
    # returns multiplier for current batch according to the standard two step schedule
    batch_idx, period = float(batch_idx), float(period)
    return decay_ratio**math.floor(batch_idx/period)

schedule = standard_schedule

print("Initialising hyperband...")
learning_rates = np.exp(rng.uniform(low=np.log(0.01), high=np.log(0.2), size=32)) # uniform samples in the log space
lr_decay_ratios = rng.uniform(low=0., high=0.5, size=32)
def get_random_config(rng):
    learning_rate = np.exp(rng.uniform(low=np.log(0.01), high=np.log(0.2)))
    lr_decay = rng.uniform(low=0., high=0.5, size=32)
    return learning_rate, lr_decay

if 'VGG' in model_tag:
    model = lambda: VGG(model_tag) # model constructor

def get_checkpoint(initial_lr, lr_decay):
    return Checkpoint(model, initial_lr, lr_decay, schedule, checkpoint_loc, log_loc)

checkpoint_handler = Hyperband

def train(checkpoints, trainloader):
    for gpu_index, checkpoint in enumerate(selected_checkpoints):
        checkpoint.init_for_epoch(gpu_index, should_update=True, epoch_size=len(trainloader))


    for batch_idx, (inputs, targets) in enumerate(trainloader):
        propagate = lambda ckpt: ckpt.propagate(inputs, targets, batch_idx, should_update=True)
        # overhead of creating executor in single thread experiment was not noticeable
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            results = executor.map(propagate, checkpoints)

        progress_str = ''
        for checkpoint in checkpoints:
            progress_str += checkpoint.progress()

        progress_bar(batch_idx, len(trainloader), progress_str)

    for checkpoint in checkpoints:
        checkpoint.epoch += 1

def validate(checkpoints, valloader, save=False):
    for gpu_index, checkpoint in enumerate(checkpoints):
        checkpoint.init_for_epoch(gpu_index, should_update=False)

    for batch_idx, (inputs, targets) in enumerate(valloader):
        propagate = lambda ckpt: ckpt.propagate(inputs, targets, batch_idx, should_update=False)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(propagate, checkpoints)

        progress_str = ''
        for checkpoint in checkpoints:
            progress_str += checkpoint.progress()
        progress_bar(batch_idx, len(valloader), progress_str)

    if save:
        for checkpoint in checkpoints:
            checkpoint.save_recent()

while True:
    # choose a subset of the candidate models and init for training and validation
    selected_checkpoints = [checkpoint_handler.get_next_checkpoint() for i in range(n_gpus)]

    # train and validate these checkpoints
    train(selected_checkpoints, trainloader)
    validate(selected_checkpoints, valloader, save=True)

    # clear checkpoints
    for checkpoint in selected_checkpoints:
        checkpoint.clear()
