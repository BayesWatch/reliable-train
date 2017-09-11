'''Class to store model specifications and training details'''
from __future__ import print_function

import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

from utils import format_filename, gridfile_parse, existing_checkpoints, \
    write_status, clean_checkpoints, get_summary_writer, format_settings_str

class Checkpoint(object):
    """
    Takes a set of hyperparameter settings, looks for existing checkpoints with
    these settings, if it finds none then initialises a checkpoint file with
    these settings. When called to train on an epoch, loads these settings and
    moves the model definition into a specified gpu.

    Trains for one epoch by being passed minibatches in a for loop, but can be
    trained at the same time as other models using parallel threads, such as
    the following in Python 3.5:

    PENDING

    After training and validation, call the save method to save the new point.
    Stores only two checkpoints at most, the most recent and the best
    performing on the validation set.

    Initialisation:
        model: a function that returns a torch.nn network, and takes no
            arguments.
        initial_lr: the initial setting of learning rate, regardless
            of schedule.
        lr_period: the period of the learning rate schedule.
        lr_schedule: a function defining the learning rate schedule, itself
            taking (period, batch_index) as arguments.
        checkpoint_loc: location to save checkpoints to.
    """
    def __init__(self, model, initial_lr, lr_period, lr_schedule, checkpoint_loc):
        # store settings
        self.initial_lr = float(initial_lr)
        self.lr_period = int(lr_period)
        # string describing settings canonically
        self.setting_str = format_settings_str(self.initial_lr, self.lr_period)
        self.checkpoint_loc = checkpoint_loc
        # store learning rate schedule
        self.lr_schedule = lambda batch_index: lr_schedule(self.lr_period, batch_index)
        
        # initialise summary writer, if we can
        self.summary_writer = get_summary_writer(args.data, [self.initial_lr, self.lr_period])

        # if checkpoint directory doesn't exist, make it
        if not os.path.isdir(checkpoint_loc):
            os.makedirs(checkpoint_loc)

        # look for checkpoints with these settings
        existing = existing_checkpoints(checkpoint_loc)
        self.best_saved = {'acc':0.0, 'not_found': False} # in case there are no saves
        self.most_recent_saved = {'epoch':0, 'not_found': False}
        for e in existing:
            print(self.setting_str, e)
            if self.setting_str == "_".join(e[0]):
                if e[1]['acc'] > self.best_saved['acc']:
                    self.best_saved = e[1]
                    print("Best now: %s"%self.best_saved['abspath'])
                if e[1]['epoch'] > self.most_recent_saved['epoch']:
                    self.most_recent_saved = e[1]

        assert False

        if self.best_saved.get('not_found', False):
            print("didn't find a save file")
            # initialise network
            self.net = model()            

            # save initialised checkpoint
            save_path = self.save(0.0, 0)
            self.best_saved = {'acc':0.0, 'abspath':save_path, 'epoch': 0}
            self.most_recent_saved = self.best_saved

            # delete network
            del self.net

    def save(self, acc, epoch):
        state = {
            'net': self.net,
            'acc': acc,
            'epoch': epoch}

        filename = format_filename(lr, period, acc, epoch)
        save_path = os.path.join(checkpoint_loc, filename)
        torch.save(state, save_path)
        return save_path 

    def load_recent(self):
        # loads most recent model
        assert False


def existing_checkpoints(checkpoint_loc):
    # should return dictionary of settings containing file locations and validation accuracies
    checkpoint_filenames = os.listdir(checkpoint_loc)
    existing_checkpoints = []
    for n in checkpoint_filenames:
        lr, period, score, epoch = parse_filename(n)
        existing_checkpoints.append(((lr, period), {'acc':float(score),
            'abspath':os.path.join(checkpoint_loc, n), 'epoch': int(epoch)}))
    return existing_checkpoints

