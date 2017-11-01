'''Class to store model specifications and training details'''
from __future__ import print_function

import math
import os

from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

import numpy as np

class Checkpoint(object):
    """
    Takes a set of hyperparameter settings, looks for existing checkpoints with
    these settings, if it finds none then initialises a checkpoint file with
    these settings. When called to train on an epoch, loads these settings and
    moves the model definition into a specified gpu.

    After training and validation, call the save method to save the new point.
    Stores only two checkpoints at most, the most recent and the best
    performing on the validation set.

    Initialisation:
        model: a function that returns a torch.nn network, and takes no
            arguments.
        initial_lr: the initial setting of learning rate, regardless
            of schedule.
        lr_decay: the size of each decay step
        minibatch_size: size of minibatch to use
        lr_schedule: a function defining the learning rate schedule, itself
            taking (period, batch_index) as arguments.
        checkpoint_loc: location to save checkpoints to.
        log_loc: location to save logs to
    """
    def __init__(self, model, initial_lr, lr_decay, minibatch_size,
                 lr_schedule, checkpoint_loc, log_loc, verbose=False,
                 multi_gpu=False, l1_factor=0., Optimizer=optim.SGD):
        self.Optimizer = Optimizer
        self.v = verbose
        self.multi_gpu = multi_gpu
        self.l1_factor = l1_factor
        # check cuda availability
        self.use_cuda = torch.cuda.is_available()

        # store settings
        self.initial_lr = float(initial_lr)
        self.lr_decay = float(lr_decay)
        self.minibatch_size = int(minibatch_size)
        # string describing settings canonically
        self.setting_str = format_settings_str(self.initial_lr, self.lr_decay, self.minibatch_size)
        self.checkpoint_loc = checkpoint_loc
        # store learning rate schedule
        self.lr_schedule = lambda batch_index, period: lr_schedule(self.lr_decay, period, batch_index)
        
        # initialise summary writer, if we can
        self.summary_writer = get_summary_writer(log_loc, [self.initial_lr, self.lr_decay, self.minibatch_size])

        # if checkpoint directory doesn't exist, make it
        if not os.path.isdir(self.checkpoint_loc):
            os.makedirs(self.checkpoint_loc)

        # look for checkpoints with these settings
        existing = existing_checkpoints(self.checkpoint_loc)
        self.best_saved = {'loss':100.0, 'not_found': True} # in case there are no saves
        self.most_recent_saved = {'epoch':0, 'not_found': True}
        for e in existing:
            if self.setting_str == "_".join(e[0]):
                if e[1]['loss'] < self.best_saved['loss']:
                    self.best_saved = e[1]
                if e[1]['epoch'] > self.most_recent_saved['epoch']:
                    self.most_recent_saved = e[1]
                    self.epoch = self.most_recent_saved['epoch']

        # count minibatches in total
        self.minibatch_idx = 0

        if not self.best_saved.get('not_found', False) and self.most_recent_saved.get('not_found', False):
            self.most_recent_saved = self.best_saved
        elif self.best_saved.get('not_found', False):
            # initialise network
            self.net = model()            

            # save initialised checkpoint
            save_path = self.save(0.0, 100.0, 0)
            self.best_saved = {'acc':0.0, 'abspath':save_path, 'epoch': 0, 'loss': 100.0}
            self.most_recent_saved = self.best_saved
            self.epoch = 0

    def save(self, acc, loss, epoch):
        state = {
            'net': self.net,
            'acc': acc,
            'loss': loss,
            'epoch': epoch}

        filename = format_filename(self.initial_lr, self.lr_decay, self.minibatch_size, acc, loss, epoch)
        save_path = os.path.join(self.checkpoint_loc, filename)
        torch.save(state, save_path)
        return save_path 

    def save_recent(self, clean=True, log=True):
        # save most recent model
        acc = 100.*self.correct/self.total
        loss = np.mean(self.accum_loss)
        old_abspath = self.most_recent_saved['abspath']
        self.most_recent_saved = {}
        self.most_recent_saved['abspath'] = self.save(acc, loss, self.epoch)
        self.most_recent_saved['acc'] = acc
        self.most_recent_saved['loss'] = loss
        self.most_recent_saved['epoch'] = self.epoch

        # change best to point to it if it's the best
        if loss < self.best_saved['loss']:
            self.best_saved = self.most_recent_saved

        if clean:
            # remove the old one if it's not the best
            if old_abspath != self.best_saved['abspath']:
                try:
                    os.remove(old_abspath)
                except:
                    import pdb
                    pdb.set_trace()

        if log:
            example_idx = self.minibatch_idx*self.minibatch_size
            self.summary_writer.add_scalar('validation/loss', loss, example_idx)
            self.summary_writer.add_scalar('validation/accuracy', acc, example_idx)

    def load_recent(self):
        # loads most recent model
        if self.v:
            print("Loading from %s"%self.most_recent_saved['abspath'])
        state = torch.load(self.most_recent_saved['abspath'])
        return state['net'], state['acc'], state['loss'], state['epoch']

    def init_for_epoch(self, gpu_index, should_update, epoch_size=None):
        self.gpu_index = gpu_index

        # load most recent params if we have none
        if 'net' not in self.__dict__:
            self.net, acc, loss, self.epoch = self.load_recent()

        if self.multi_gpu and not isinstance(self.net, torch.nn.DataParallel):
            self.net.cuda(self.gpu_index)
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
        elif not self.multi_gpu:
            self.net.cuda(self.gpu_index)

        # always set up criterion and optimiser
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.Optimizer(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
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

            self.sparsify = lambda x: self.optimizer.maybe_sparsify(x+current_batch, 60*epoch_size)
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
            if self.l1_factor > 1e-8:
                reg_loss = 0
                for param in self.net.parameters() if not isinstance(self.net,
                        torch.nn.DataParallel) else self.net.module.parameters():
                    reg_loss += self.l1_factor*l1_loss(param)
                reg_loss.backward()
            self.optimizer.step()
            self.minibatch_idx += 1

        loss = loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        self.total += targets.size(0)
        self.correct += predicted.eq(targets.data).cpu().sum()
        self.accum_loss.append(loss)

        if should_update:
            acc = 100.*self.correct/self.total
            example_idx = self.minibatch_idx*self.minibatch_size
            self.summary_writer.add_scalar('train/loss', loss, example_idx)
            self.summary_writer.add_scalar('train/accuracy', acc, example_idx)
            self.summary_writer.add_scalar('train/learning_rate', lr, example_idx)
            net_sparsity = sparsity(self.net if not isinstance(self.net,
                                    torch.nn.DataParallel) else self.net.module)
            self.summary_writer.add_scalar('train/sparsity', net_sparsity, example_idx)
            # if we've reached a high enough batch then sparsify
            if hasattr(self.optimizer, 'sparsify'):
                self.sparsify(batch_index)

        return  loss

    def progress(self):
        acc = 100.*self.correct/self.total
        net_sparsity = sparsity(self.net if not isinstance(self.net,
            torch.nn.DataParallel) else self.net.module)
        return '| %i | %.3f | %.3f%% | %.3f |'\
            % (self.epoch, np.mean(self.accum_loss), acc, net_sparsity)

    def clear(self):
        del self.net

    def __repr__(self):
        return self.setting_str


def existing_checkpoints(checkpoint_loc):
    # should return dictionary of settings containing file locations and validation accuracies
    checkpoint_filenames = os.listdir(checkpoint_loc)
    checkpoint_filenames = [cf for cf in checkpoint_filenames if '.t7' in cf]
    existing = []
    for n in checkpoint_filenames:
        lr, decay, minibatch_size, acc, loss, epoch = n[:-3].split("_")
        existing.append(((lr, decay, minibatch_size), {'acc':float(acc), 'loss':float(loss),
            'abspath':os.path.join(checkpoint_loc, n), 'epoch': int(epoch)}))
    return existing


def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def format_filename(lr, decay, minibatch_size, acc, loss, epoch):
    fname_string = format_settings_str(lr, decay, minibatch_size, acc, loss, epoch)
    return fname_string+".t7"


def format_settings_str(*settings):
    str_components = []
    for s in settings:
        if type(s) is float:
            str_components.append("%06.3f"%s)
        elif type(s) is int:
            str_components.append("%05d"%s)
        elif isinstance(s, np.float):
            str_components.append("%06.3f"%s)
        else:
            raise ValueError("%s of type %s is not a valid entry"%(s, type(s)))
    return "_".join(str_components)


try:
    from tensorboardX import SummaryWriter
    def get_summary_writer(log_loc, settings):
        # save to subdir describing the hyperparam settings
        dirname = format_settings_str(*settings)
        return SummaryWriter(os.path.join(log_loc, dirname))
except ImportError:
    print("tensorboard-pytorch not detected, will not write plot logs anywhere")
    class DummyWriter(object):
        def __init__(self, log_dir):
            return None
        def add_scalar(self, tag, scalar_value, global_step):
            return None
    def get_summary_writer(log_loc, settings):
        return DummyWriter(log_loc)

def l1_loss(x):
    return torch.abs(x).sum()

def sparsity(model):
    total = 0
    active = 0
    for param in model.parameters():
        active += (torch.abs(param) > 1e-3).float().sum().data.cpu().numpy()
        total += reduce(lambda a,b: a*b, param.size()) # no I couldn't think of a better way
    return float((active/total))

if __name__ == '__main__':
    # test for the sparsity bit, nothing else
    from models import *
    net = ResNet50()
    print(sparsity(net))
