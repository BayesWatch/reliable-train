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
import traceback

import torch
import torch.nn as nn
import torch.nn.init as init

import numpy as np


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

def format_l1(l1):
    return (".l1_%.01E"%l1).lower().replace("0", "")

def format_l2(l1):
    return (".l2_%.01E"%l1).lower().replace("0", "")

def sigterm_handler(_signo, _stack_frame):
    sigterm_trace = "    ".join(traceback.format_stack()[:-1])
    cmdline = " ".join(sys.argv)
    logging.info("COMMAND RECEIVED SIGTERM: %s was at line:\n"%cmdline+sigterm_trace)
    sys.exit(1)

def cleanup(checkpoint_loc, log_loc):
    are_you_sure = input("Deleting ALL EXPERIMENTS in %s, and ALL LOGS in %s is that OK? (y/n)"%(checkpoint_loc, log_loc))
    if are_you_sure == 'y':
        import shutil
        def try_delete(loc):
            try:
                shutil.rmtree(loc)
            except FileNotFoundError:
                print("%s already clean, nothing to delete"%loc)
        print("Deleting checkpoints in %s..."%checkpoint_loc)
        try_delete(checkpoint_loc)
        print("Deleting logs in %s..."%log_loc)
        try_delete(log_loc)
    else:
        assert False

def sgdr(decay_ratio, period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    # I'm sorry decay_ratio is never used, I'm just hacking this thing together
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx/restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(batch_idx/restart_period)
    return 0.5*(1.0 + math.cos(radians))

def standard_schedule(decay_ratio, period, batch_idx):
    # returns multiplier for current batch according to the standard two step schedule
    batch_idx, period = float(batch_idx), float(period)
    return decay_ratio**math.floor(batch_idx/period)

