'''Test best model on CIFAR10.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision

import sys
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

from models import *
from torch.autograd import Variable

import numpy as np

from utils import gridfile_parse, write_status
from checkpoint import Checkpoint, existing_checkpoints
from data import cifar10
from main import validate

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--data', '-d', default=os.environ.get('SCRATCH',os.getcwd()), help='place to store data')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

_, _, testloader = cifar10(args.args.data)

model_tag = 'VGG16'
checkpoint_loc = os.path.join(args.data, 'checkpoint', model_tag)
print("Checkpoints loading from: %s"%checkpoint_loc)

# find best checkpoint, load and run on all gpus
existing = existing_checkpoints(checkpoint_loc)
best_acc = 0.0
best_settings = None
best_saved = None
for e in existing:
    if e[1]['acc'] > best_acc:
        best_saved = e[1]
        best_settings = e[0]
        print("Best so far: %s %s"%("_".join(e[0]), best_saved['abspath']))
state = torch.load(best_saved['abspath'])
net, acc, epoch = state['net'], state['acc'], state['epoch']
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

print("Testing...")
# train and validate these checkpoints
validate(selected_checkpoints, testloader, save=False)
