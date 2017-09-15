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

import numpy as np

from utils import format_filename, gridfile_parse, existing_checkpoints, \
    write_status, clean_checkpoints, get_summary_writer, format_settings_str, parse_filename

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
        lr_decay: the size of each decay step
        lr_schedule: a function defining the learning rate schedule, itself
            taking (period, batch_index) as arguments.
        checkpoint_loc: location to save checkpoints to.
        log_loc: location to save logs to
    """
    def __init__(self, model, initial_lr, lr_decay, lr_schedule, checkpoint_loc, log_loc):
        # check cuda availability
        self.use_cuda = torch.cuda.is_available()

        # store settings
        self.initial_lr = float(initial_lr)
        self.lr_decay = float(lr_decay)
        # string describing settings canonically
        self.setting_str = format_settings_str(self.initial_lr, self.lr_decay)
        self.checkpoint_loc = checkpoint_loc
        # store learning rate schedule
        self.lr_schedule = lambda batch_index, period: lr_schedule(self.lr_decay, period, batch_index)
        
        # initialise summary writer, if we can
        self.summary_writer = get_summary_writer(log_loc, [self.initial_lr, self.lr_decay])

        # if checkpoint directory doesn't exist, make it
        if not os.path.isdir(self.checkpoint_loc):
            os.makedirs(self.checkpoint_loc)

        # look for checkpoints with these settings
        existing = existing_checkpoints(self.checkpoint_loc)
        self.best_saved = {'acc':0.0, 'not_found': True} # in case there are no saves
        self.most_recent_saved = {'epoch':0, 'not_found': True}
        for e in existing:
            if self.setting_str == "_".join(e[0]):
                if e[1]['acc'] > self.best_saved['acc']:
                    self.best_saved = e[1]
                    print("Best for %s: %s"%(self.setting_str, self.best_saved['abspath']))
                if e[1]['epoch'] > self.most_recent_saved['epoch']:
                    self.most_recent_saved = e[1]

        # count minibatches in total
        self.minibatch_idx = 0

        if not self.best_saved.get('not_found', False) and self.most_recent_saved.get('not_found', False):
            self.most_recent_saved = self.best_saved
        elif self.best_saved.get('not_found', False):
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

        filename = format_filename(self.initial_lr, self.lr_decay, acc, epoch)
        save_path = os.path.join(self.checkpoint_loc, filename)
        torch.save(state, save_path)
        return save_path 

    def save_recent(self, clean=True, log=True):
        # save most recent model
        acc = 100.*self.correct/self.total
        old_abspath = self.most_recent_saved['abspath']
        self.most_recent_saved = {}
        self.most_recent_saved['abspath'] = self.save(acc, self.epoch)
        self.most_recent_saved['acc'] = 0.0
        self.most_recent_saved['epoch'] = self.epoch

        # change best to point to it if it's the best
        if acc > self.best_saved['acc']:
            self.best_saved = self.most_recent_saved

        if clean:
            # remove the old one if it's not the best
            if old_abspath != self.best_saved['abspath']:
                os.remove(old_abspath)

        if log:
            self.summary_writer.add_scalar(self.setting_str + '/validation/loss', np.mean(self.accum_loss), self.minibatch_idx)
            self.summary_writer.add_scalar(self.setting_str + '/validation/accuracy', acc, self.minibatch_idx)

    def load_recent(self):
        # loads most recent model
        state = torch.load(self.most_recent_saved['abspath'])
        return state['net'], state['acc'], state['epoch']

    def init_for_epoch(self, gpu_index, should_update, epoch_size=None):
        self.gpu_index = gpu_index

        # load most recent params if we have none
        if 'net' not in self.__dict__:
            self.net, acc, self.epoch = self.load_recent()
        self.net.cuda(self.gpu_index)

        # always set up criterion and optimiser
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.correct, self.total, self.accum_loss = 0, 0, []

        if should_update:
            # set into training mode
            self.net.train()

            # set up learning rate callback
            current_batch = epoch_size * self.epoch
            period = 60*epoch_size
            self.lr_schedule_callback = lambda x: self.initial_lr*self.lr_schedule(x+current_batch, period)

            self.epoch_size = epoch_size

            self.minibatch_idx = current_batch
        else:
            self.net.eval()

    def train(self, mode=True):
        return self.net.train()

    def eval(self):
        return self.train(False)

    def propagate(self, inputs, targets, batch_index, should_update):
        # puts inputs and targets on correct gpu
        # sets the learning rate
        # forward pass
        # backward pass and update if should_update is true
        # records no. correctly classified and total
        # records loss

        if not should_update:
            if self.net.training:
                raise ValueError("Model should not be in train mode if not updating parameters.")

        if self.use_cuda:
            inputs, targets = inputs.cuda(self.gpu_index), targets.cuda(self.gpu_index)
        if should_update:
            lr = self.lr_schedule_callback(batch_index)
            self.optimizer = set_optimizer_lr(self.optimizer, lr)
            self.optimizer.zero_grad()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)

        if should_update:
            loss.backward()
            self.optimizer.step()
            self.minibatch_idx += 1

        loss = loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        self.total += targets.size(0)
        self.correct += predicted.eq(targets.data).cpu().sum()
        self.accum_loss.append(loss)

        if should_update:
            acc = 100.*self.correct/self.total
            self.summary_writer.add_scalar(self.setting_str + '/train/loss', loss, self.minibatch_idx)
            self.summary_writer.add_scalar(self.setting_str + '/train/accuracy', acc, self.minibatch_idx)

        return  loss

    def progress(self):
        acc = 100.*self.correct/self.total
        return '| %.3f | %.3f%% |'\
            % (np.mean(self.accum_loss), acc)

    def clear(self):
        del self.net

    def __repr__(self):
        return self.setting_str

def existing_checkpoints(checkpoint_loc):
    # should return dictionary of settings containing file locations and validation accuracies
    checkpoint_filenames = os.listdir(checkpoint_loc)
    existing_checkpoints = []
    for n in checkpoint_filenames:
        lr, decay, acc, epoch = parse_filename(n)
        existing_checkpoints.append(((lr, decay), {'acc':float(acc),
            'abspath':os.path.join(checkpoint_loc, n), 'epoch': int(epoch)}))
    return existing_checkpoints

def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

