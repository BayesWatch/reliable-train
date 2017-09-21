'''Train CIFAR10.'''
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

from utils import gridfile_parse, write_status, \
        clean_checkpoints, get_summary_writer, progress_bar
from checkpoint import Checkpoint
from hyperband import Hyperband
from data import cifar10

from itertools import combinations

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data', '-d', default=os.environ.get('SCRATCH',os.getcwd()), help='place to store data')
#parser.add_argument('--sgdr', action='store_true', help='use the SGDR learning rate schedule')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

n_gpus = torch.cuda.device_count()

trainloader, valloader, _ = cifar10(args.data)

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
def get_random_config(rng):
    learning_rate = np.exp(rng.uniform(low=np.log(0.01), high=np.log(0.2)))
    lr_decay = rng.uniform(low=0., high=0.5)
    return learning_rate, lr_decay

if 'VGG' in model_tag:
    model = lambda: VGG(model_tag) # model constructor

def get_checkpoint(initial_lr, lr_decay):
    return Checkpoint(model, initial_lr, lr_decay, schedule, checkpoint_loc, log_loc)

checkpoint_handler = Hyperband(get_random_config)

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

def validate(checkpoints, loader, save=False):
    for gpu_index, checkpoint in enumerate(checkpoints):
        checkpoint.init_for_epoch(gpu_index, should_update=False)

    for batch_idx, (inputs, targets) in enumerate(loader):
        propagate = lambda ckpt: ckpt.propagate(inputs, targets, batch_idx, should_update=False)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(propagate, checkpoints)

        progress_str = ''
        for checkpoint in checkpoints:
            progress_str += checkpoint.progress()
        progress_bar(batch_idx, len(loader), progress_str)

    if save:
        for checkpoint in checkpoints:
            checkpoint.save_recent()

    try:
        return [c.best_saved['loss'] for c in checkpoints]
    except:
        import pdb
        pdb.set_trace()

print("Beginning search...")
leftover = []
while True:
    # choose a subset of the candidate models and init for training and validation
    checkpoint_handler.populate_queue(n_gpus)
    selected_checkpoints, idxs = [], []
    for idx, settings in checkpoint_handler.queue:
        idxs.append(idx)
        selected_checkpoints.append(get_checkpoint(*settings))
    checkpoint_handler.queue = []

    # train and validate these checkpoints
    train(selected_checkpoints, trainloader)
    losses = validate(selected_checkpoints, valloader, save=True)

    # update losses
    checkpoint_handler.update(losses, idxs)

    # clear checkpoints
    for checkpoint in selected_checkpoints:
        try:
            checkpoint.clear()
        except:
            import pdb
            pdb.set_trace()
    # then destroy them (paranoid, because clear should happen when we del)
    del selected_checkpoints
