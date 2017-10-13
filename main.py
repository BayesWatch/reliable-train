'''Train on CIFAR10 persistently (script can stop and start as much as you
want, *it'll keep trying*).'''
from __future__ import print_function

import torch

import sys
import os
import argparse
import traceback
import logging

from models import *
from torch.autograd import Variable

import numpy as np

from utils import ProgressBar, format_l1
from checkpoint import Checkpoint
from data import cifar10
from seppuku import exit_after

from itertools import combinations

def parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training\nlearning rate will decay every 60 epochs')
    parser.add_argument('--scratch', '-s', default=os.environ.get('SCRATCH',os.getcwd()), help='place to store data')
    parser.add_argument('--lr', default=0.1, help='learning rate')
    parser.add_argument('--l1', default=0., type=float, help='l1 regularisation factor')
    parser.add_argument('--lr_decay', default=0.01, help='learning rate decay coefficient')
    parser.add_argument('--minibatch', '-M', default=128, help='minibatch size')
    parser.add_argument('--epochs', '-N', default=180, help='number of epochs to train for')
    parser.add_argument('--gpu', default=0, help='index of gpu to use')
    parser.add_argument('--multi_gpu', action='store_true', help='use all available gpus')
    parser.add_argument('--model', default='resnet50', type=str, help='string to choose model')
    parser.add_argument('--model_multiplier', default=4, type=int, help='multiplier for number of planes in model')
    parser.add_argument('-v', action='store_true', help='verbose with progress bar')
    parser.add_argument('--evaluate', action='store_true', help='run on test set')
    #parser.add_argument('--sgdr', action='store_true', help='use the SGDR learning rate schedule')
    args = parser.parse_args()
    return args

def format_model_tag(model, model_multiplier, l1):
    if 'resnet' in model:
        model_tag = model+".%02d"%model_multiplier+format_l1(l1)
    else:
        model_tag = model
    return model_tag

def main(args):
    if args.v:
        progress_bar = ProgressBar()

    use_cuda = torch.cuda.is_available()
    gpu_index = int(args.gpu) if not args.multi_gpu else None
    best_acc = 0  # best test accuracy

    n_gpus = torch.cuda.device_count()

    trainloader, valloader, testloader = cifar10(args.scratch, args.minibatch, verbose=args.v)

    # Set where to save and load checkpoints, use model_tag for directory name
    model_tag = format_model_tag(args.model, args.model_multiplier, args.l1)
    checkpoint_loc = os.path.join(args.scratch, 'checkpoint', model_tag)
    if not os.path.isdir(checkpoint_loc):
        os.makedirs(checkpoint_loc)
    if args.v:
        print("Checkpoint saved to: %s"%checkpoint_loc)

    # Set where to append tensorboard logs
    log_loc = os.path.join(args.scratch, 'logs', model_tag)
    if not os.path.isdir(log_loc):
        os.makedirs(log_loc)
    if args.v:
        print("Tensorboard logs saved to: %s"%log_loc)

    # define learning rate schedule and make it the default
    # generalised so it would work with any schedule definable with mbatch index
    def standard_schedule(decay_ratio, period, batch_idx):
        # returns multiplier for current batch according to the standard two step schedule
        batch_idx, period = float(batch_idx), float(period)
        return decay_ratio**math.floor(batch_idx/period)
    schedule = standard_schedule

    # complicated way to initialise the Checkpoint object that'll hold our
    # model
    if 'VGG' in model_tag:
        model = lambda: VGG(model_tag) # model constructor
    elif 'resnet' in model_tag:
        if '50' in model_tag:
            model = lambda: ResNet50(args.model_multiplier)
    elif 'butterfly' in model_tag:
        if 'res' in model_tag:
            model = lambda: ButterflyResNet18()
        else:
            model = lambda: ButterflyNet()
    elif 'mobilenet' in model_tag:
        model = lambda: MobileNet()
    def get_checkpoint(initial_lr, lr_decay, minibatch_size):
        return Checkpoint(model, initial_lr, lr_decay, minibatch_size,
                schedule, checkpoint_loc, log_loc, verbose=args.v,
                multi_gpu=args.multi_gpu, l1_factor=args.l1)
    checkpoint = get_checkpoint(args.lr, args.lr_decay, args.minibatch)

    #@exit_after(240)
    def train(checkpoint, trainloader):
        checkpoint.init_for_epoch(gpu_index, should_update=True, epoch_size=len(trainloader))

        batch_idx = 0
        for inputs, targets in trainloader:
            batch_idx += 1
            checkpoint.propagate(inputs, targets, batch_idx, should_update=True)

            if args.v:
                progress_str = ''
                progress_str += checkpoint.progress()
                progress_bar(batch_idx, len(trainloader), progress_str)
        checkpoint.epoch += 1
        return None
    
    #@exit_after(240)
    def validate(checkpoint, loader, save=False):
        checkpoint.init_for_epoch(gpu_index, should_update=False)

        batch_idx = 0
        for inputs, targets in loader:
            batch_idx += 1
            checkpoint.propagate(inputs, targets, batch_idx, should_update=False)

            if args.v:
                progress_str = ''
                progress_str += checkpoint.progress()
                progress_bar(batch_idx, len(loader), progress_str)

        if save:
            checkpoint.save_recent()
        return None

    for epoch in range(int(args.epochs) - checkpoint.epoch):
        # train and validate this checkpoint
        if not args.evaluate:
            train(checkpoint, trainloader)
            validate(checkpoint, valloader, save=True)
        else:
            validation_loss = checkpoint.best_saved['loss']
            validation_acc = checkpoint.best_saved['acc']
            validate(checkpoint, testloader, save=False)
            test_loss = np.mean(checkpoint.accum_loss)
            test_acc = 100.*checkpoint.correct/checkpoint.total
            print("Validation loss: %.3f\n"%validation_loss+
                  "Validation accuracy: %.3f\n"%validation_acc+
                  "Test loss: %.3f\n"%test_loss+
                  "Test accuracy: %.3f"%test_acc)
            return None
    print(checkpoint.most_recent_saved['loss'])

if __name__ == '__main__':
    args = parse()
    # initialise logging
    model_tag = format_model_tag(args.model, args.model_multiplier, args.l1)
    logging_loc = os.path.join(args.scratch, 'checkpoint', model_tag, 'errors.log')
    if not os.path.isdir(os.path.dirname(logging_loc)):
        os.makedirs(os.path.dirname(logging_loc))
    logging.basicConfig(filename=logging_loc, level=logging.DEBUG)
    # useful for logging
    cmdline = "python " + " ".join(sys.argv)
    try:
        main(args)
    except Exception as e:
        # log the error then raise the exception again
        logging.info("COMMAND FAILED: %s"%cmdline)
        logging.error(traceback.format_exc())
        raise
    except KeyboardInterrupt:
        logging.info("COMMAND DIED HONOURABLY: %s"%cmdline)
    except:
        logging.info("COMMAND DIED MYSTERIOUSLY: %s"%cmdline)
