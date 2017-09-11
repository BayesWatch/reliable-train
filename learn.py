'''Training and validation functions'''
from __future__ import print_function

import math
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

from utils import progress_bar

def unpack_ckpt(checkpoint, gpu_idx):
    use_cuda = torch.cuda.is_available()
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    if use_cuda:
        net.cuda(gpu_idx)
        cudnn.benchmark = True

    return net, best_acc, start_epoch

def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def sgdr(period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx/restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(batch_idx/restart_period)
    return 0.5*(1.0 + math.cos(radians))

def standard_schedule(period, batch_idx):
    # returns multiplier for current batch according to the standard two step schedule
    batch_idx, period = float(batch_idx), float(period)
    return 0.1**math.floor(batch_idx/period)

# Training
def train(checkpoints, trainloader, lr_schedule):
    use_cuda = torch.cuda.is_available()
    for gpu_idx, checkpoint in enumerate(checkpoints):
        net, best_acc, start_epoch = unpack_ckpt(checkpoint, gpu_idx)
        net.train()

        # set up learning rate callback
        current_batch = len(trainloader) * start_epoch
        lr = float(checkpoint['init_lr'])
        period = checkpoint['period']*len(trainloader)
        checkpoint['lr_schedule_callback'] = lambda x: lr*lr_schedule(period, x+current_batch)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        checkpoint.update({'criterion': criterion, 'optimizer':optimizer, 'gpu_idx':gpu_idx})
        checkpoint['correct'] = 0
        checkpoint['total'] = 0
        checkpoint['avg_train_loss'] = 0.0

    with ThreadPoolExecutor(max_workers=4) as executor:
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            propagate_this = lambda ckpt: propagate(ckpt, inputs, targets, batch_idx, update=True)
            results = executor.map(propagate_this, checkpoints)

            progress_str = ''
            for checkpoint, train_loss in results:
                correct, total, recent_lr = checkpoint['correct'], checkpoint['total'], checkpoint['recent_lr']
                checkpoint['train_acc'] = 100.*correct/total
                checkpoint['avg_train_loss'] += train_loss
                progress_str += '| Loss: %.3f | Acc: %.3f%% (%d/%d) |'\
                    % (train_loss, checkpoint['train_acc'], correct, total)

            progress_bar(batch_idx, len(trainloader), progress_str)

    for checkpoint in checkpoints:
        checkpoint['epoch'] += 1
        checkpoint['avg_train_loss'] = checkpoint['avg_train_loss']/len(trainloader)
        checkpoint['summary_writer'].add_scalar('train_loss', checkpoint['avg_train_loss'], checkpoint['epoch'])
        checkpoint['summary_writer'].add_scalar('train_accuracy', checkpoint['train_acc'], checkpoint['epoch'])
        checkpoint['summary_writer'].add_scalar('learning_rate', checkpoint['recent_lr'], checkpoint['epoch'])
    # return nothing because we're doing epochs in place, urgh
    return None

def propagate(checkpoint, inputs, targets, batch_idx, update=False):
    gpu_idx = checkpoint['gpu_idx']
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        inputs, targets = inputs.cuda(gpu_idx), targets.cuda(gpu_idx)
    if update:
        checkpoint['recent_lr'] = checkpoint['lr_schedule_callback'](batch_idx)
        checkpoint['optimizer'] = set_optimizer_lr(checkpoint['optimizer'], checkpoint['recent_lr'])
        checkpoint['optimizer'].zero_grad()

    inputs, targets = Variable(inputs), Variable(targets)
    outputs = checkpoint['net'](inputs)
    loss = checkpoint['criterion'](outputs, targets)

    if update:
        loss.backward()
        checkpoint['optimizer'].step()

    loss = loss.data[0]
    _, predicted = torch.max(outputs.data, 1)
    checkpoint['total'] += targets.size(0)
    checkpoint['correct'] += predicted.eq(targets.data).cpu().sum()

    return checkpoint, loss

def validate(checkpoints, valloader, checkpoint_loc, save=False):
    use_cuda = torch.cuda.is_available()
    for gpu_idx, checkpoint in enumerate(checkpoints):
        net, best_acc, start_epoch = unpack_ckpt(checkpoint, gpu_idx)

        criterion = nn.CrossEntropyLoss()

        net.eval()

        checkpoint['gpu_idx'] = gpu_idx
        checkpoint['correct'] = 0
        checkpoint['total'] = 0
        checkpoint['avg_val_loss'] = 0.0

    with ThreadPoolExecutor(max_workers=4) as executor:
        for batch_idx, (inputs, targets) in enumerate(valloader):

            propagate_this = lambda ckpt: propagate(ckpt, inputs, targets, batch_idx, update=False)
            results = executor.map(propagate_this, checkpoints)

            progress_str = ''
            for checkpoint, test_loss in results:
                correct, total = checkpoint['correct'], checkpoint['total']
                checkpoint['val_acc'] = 100.*correct/total
                progress_str += '| Loss: %.3f | Acc: %.3f%% (%d/%d) |'\
                    % (test_loss, checkpoint['val_acc'], correct, total)
                checkpoint['avg_val_loss'] += test_loss
            progress_bar(batch_idx, len(valloader), progress_str)

    for checkpoint in checkpoints:
        correct, total = checkpoint['correct'], checkpoint['total']
        checkpoint['avg_val_loss'] = checkpoint['avg_val_loss']/len(valloader)
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = save_checkpoint(checkpoint_loc, net, acc, checkpoint['epoch'], checkpoint['init_lr'], checkpoint['period'])
            checkpoint['recent_abspath'] = state['recent_abspath']
            best_acc = acc
        checkpoint['summary_writer'].add_scalar('train_loss', checkpoint['avg_val_loss'], checkpoint['epoch'])
        checkpoint['summary_writer'].add_scalar('train_accuracy', checkpoint['val_acc'], checkpoint['epoch'])

