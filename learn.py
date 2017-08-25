'''Training and validation functions'''
from __future__ import print_function

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

def save_checkpoint(checkpoint_loc, net, acc, epoch):
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
    }

    if not os.path.isdir(checkpoint_loc):
        os.mkdir(checkpoint_loc)
    torch.save(state, os.path.join(checkpoint_loc,'ckpt.t7'))
    return state

# Training
def train(epoch, checkpoints, trainloader, lr):
    use_cuda = torch.cuda.is_available()
    for gpu_idx, checkpoint in enumerate(checkpoints):
        net, best_acc, start_epoch = unpack_ckpt(checkpoint, gpu_idx)
        net.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        checkpoint.update({'criterion': criterion, 'optimizer':optimizer, 'gpu_idx':gpu_idx})
        checkpoint['correct'] = 0
        checkpoint['total'] = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        with ThreadPoolExecutor(max_workers=4) as executor:
            propagate_this = lambda ckpt: propagate(ckpt, inputs, targets)
            results = executor.map(propagate_this, checkpoints)
        #results = []
        #for checkpoint in checkpoints:
        #    checkpoint, train_loss = propagate(checkpoint, inputs, targets)
        #    results.append((checkpoint, train_loss))
        progress_str = ''
        for checkpoint, train_loss in results:
            correct, total = checkpoint['correct'], checkpoint['total']
            progress_str += '| Loss: %.3f | Acc: %.3f%% (%d/%d) |'\
                % (train_loss, 100.*correct/total, correct, total)
        progress_bar(batch_idx, len(trainloader), progress_str)
    return None

def propagate(checkpoint, inputs, targets):
    gpu_idx = checkpoint['gpu_idx']
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        inputs, targets = inputs.cuda(gpu_idx), targets.cuda(gpu_idx)
    checkpoint['optimizer'].zero_grad()
    inputs, targets = Variable(inputs), Variable(targets)
    outputs = checkpoint['net'](inputs)
    loss = checkpoint['criterion'](outputs, targets)
    loss.backward()
    checkpoint['optimizer'].step()

    train_loss = loss.data[0]
    _, predicted = torch.max(outputs.data, 1)
    checkpoint['total'] += targets.size(0)
    checkpoint['correct'] += predicted.eq(targets.data).cpu().sum()

    return checkpoint, train_loss

def validate(epoch, checkpoints, valloader, checkpoint_loc):
    use_cuda = torch.cuda.is_available()
    net, best_acc, start_epoch = unpack_ckpt(checkpoint, gpu_idx)

    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valloader):
        if use_cuda:
            inputs, targets = inputs.cuda(gpu_idx), targets.cuda(gpu_idx)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        #progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        save_checkpoint(checkpoint_loc, net, acc, epoch+start_epoch)
        best_acc = acc
    return best_acc

def test(epoch):
    use_cuda = torch.cuda.is_available()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

