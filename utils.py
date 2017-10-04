'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import re

import torch
import torch.nn as nn
import torch.nn.init as init

import numpy as np


def get_num_gen(gen):
    return sum(1 for x in gen)

def flops_layer(layer, w, h):
    """
    Calculate the number of flops for given a string information of layer.
    """
    idx_type_end = layer.find('(')

    params = re.findall('[^a-z](\d+)', layer)
    flops = 1

    if layer.find('Linear') >= 0:
        C1 = int(params[0])
        C2 = int(params[1])
        flops = C1 * C2

    elif layer.find('Conv2d') >= 0:

        matches = re.findall('([A-z_]+)[ ]*=(\([0-9]+,[ ]*[0-9]+\))', layer)
        padding = [0,0]
        stride = [0,0]
        dilation = [1,1]

        for m, _ in enumerate(matches):
            if matches[m][0]=='padding':
                padding[0] = int(re.findall('[^a-z](\d+)',matches[m][1])[0])
                padding[1] = int(re.findall('[^a-z](\d+)',matches[m][1])[1])
            elif matches[m][0]=='stride':
                stride[0] = int(re.findall('[^a-z](\d+)',matches[m][1])[0])
                stride[1] = int(re.findall('[^a-z](\d+)',matches[m][1])[1])
            elif matches[m][0]=='dilation':
                dilation[0] = int(re.findall('[^a-z](\d+)',matches[m][1])[0])
                dilation[1] = int(re.findall('[^a-z](\d+)',matches[m][1])[1])

        C1 = int(params[0])
        C2 = int(params[1])
        K1 = int(params[2])
        K2 = int(params[3])

        # image size
        flops = C1 * C2 * K1 * K2 * h * w
        h = math.floor((h+2*padding[0]-dilation[0]*(K1-1)-1)/stride[0]+1)
        w = math.floor((w+2*padding[1]-dilation[1]*(K2-1)-1)/stride[1]+1)


    elif layer.find('MaxPool2d') >= 0:

        matches = re.findall('([A-z_]+)[ ]*=(\([0-9]+,[ ]*[0-9]+\))', layer)

        size = [0,0]
        stride = [0,0]
        dilation = [0,0]
        padding = [0,0]

        for m, _ in enumerate(matches):
            if matches[m][0]=='size':
                size[0] = int(re.findall('[^a-z](\d+)',matches[m][1])[0])
                size[1] = int(re.findall('[^a-z](\d+)',matches[m][1])[1])
            elif matches[m][0]=='stride':
                stride[0] = int(re.findall('[^a-z](\d+)',matches[m][1])[0])
                stride[1] = int(re.findall('[^a-z](\d+)',matches[m][1])[1])
            elif matches[m][0]=='dilation':
                dilation[0] = int(re.findall('[^a-z](\d+)',matches[m][1])[0])
                dilation[1] = int(re.findall('[^a-z](\d+)',matches[m][1])[1])
            elif matches[m][0]=='padding':
                padding[0] = int(re.findall('[^a-z](\d+)',matches[m][1])[0])
                padding[1] = int(re.findall('[^a-z](\d+)',matches[m][1])[1])

        h = math.floor((h+2*padding[0]-dilation[0]*(size[0]-1)-1)/stride[0]+1)
        w = math.floor((w+2*padding[1]-dilation[1]*(size[1]-1)-1)/stride[1]+1)

    return flops, w, h


def calculate_flops(gen, w, h):
    # I have modified this function so that it keeps track of the width/height of the representation.
    # net.children() goes in as gen
    flops = 0

    for child in gen:
        num_children = get_num_gen(child.children())

        # leaf node
        if num_children == 0:
            no_flops, w, h = flops_layer(str(child), w, h,)
            flops += no_flops

        else:
            no_flops,w ,h = calculate_flops(child.children(),w,h)
            flops += no_flops
            #print(flops)
    return flops, w, h

def get_no_params(net):

    params = net.state_dict()
    tot=0
    for p in params:
        no = params[p].view(-1).__len__()
        tot += no
        print('%s has %d params' % (p,no))
    print('Net has %d params in total' % tot)
    return tot


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

class ProgressBar(object):
    def __init__(self):
        try:
            _, term_width = os.popen('stty size', 'r').read().split()
            term_width = int(term_width)
        except:
            term_width = 80

        TOTAL_BAR_LENGTH = 65.
        self.__dict__.update(locals())
        self.last_time = time.time()
        self.begin_time = self.last_time
        
    def __call__(self, current, total, msg=None):
        TOTAL_BAR_LENGTH = self.TOTAL_BAR_LENGTH
        term_width = self.term_width
        if current == 0:
            self.begin_time = time.time()  # Reset for new bar.

        cur_len = int(TOTAL_BAR_LENGTH*current/total)
        rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

        sys.stdout.write(' [')
        for i in range(cur_len):
            sys.stdout.write('=')
        sys.stdout.write('>')
        for i in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write(']')

        cur_time = time.time()
        step_time = cur_time - self.last_time
        self.last_time = cur_time
        tot_time = cur_time - self.begin_time

        L = []
        L.append('  Step: %s' % format_time(step_time))
        L.append(' | Tot: %s' % format_time(tot_time))
        if msg:
            L.append(' | ' + msg)

        msg = ''.join(L)
        sys.stdout.write(msg)
        for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
            sys.stdout.write(' ')

        # Go back to the center of the bar.
        for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
            sys.stdout.write('\b')
        sys.stdout.write(' %d/%d ' % (current+1, total))

        if current < total-1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def gridfile_parse(file_object):
    for line in file_object:
        if 'learning rate' in line:
            lr_settings = line.split(',')[1:]
        elif 'schedule period' in line:
            period_settings = line.split(',')[1:]
    # return enumerated combinations
    combinations = []
    for lr in lr_settings:
        for period in period_settings:
            combinations.append(("%06.3f"%float(lr), "%05d"%int(period)))
    return combinations

def write_status(log_filename, checkpoint_loc, sgdr_or_not):
    print("Writing to log file...")
    with open("grid_%s.csv"%("sgdr" if sgdr_or_not else "default"), "r") as f:
        grid_settings = gridfile_parse(f)
    existing = existing_checkpoints(checkpoint_loc)
    log_table = ""
    for i, setting in enumerate(grid_settings):
        if i == 0:
            prev_lr = setting[0]
        elif setting[0] != prev_lr:
            log_table += "\n"
            prev_lr = setting[0]
        try:
            filename = os.path.split(existing[setting]['abspath'])[-1]
            acc = float(existing[setting]['acc'])
        except KeyError:
            filename = format_filename(float(setting[0]), float(setting[1]), 0.0)
            acc = 0.0
        log_table += "%s:%06.3f    "%(filename, acc)

    with open(log_filename, 'w') as f:
        f.write(log_table)

def format_l1(l1):
    return (".l1_%.01E"%l1).lower().replace("0", "")
