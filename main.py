'''Train on CIFAR10 persistently (script can stop and start as much as you
want, *it'll keep trying*).'''
from __future__ import print_function

import torch
import torch.nn as nn
from torch import optim

import sys
import os
import argparse
import traceback
import logging

from models import *
from torch.autograd import Variable

import numpy as np

from utils import ProgressBar, format_l1, format_l2, sigterm_handler, standard_schedule, sgdr
from checkpoint import Checkpoint, format_settings_str
from data import cifar10

from itertools import combinations

import signal

# if we receive SIGTERM, we want to log what we were doing when that happened
signal.signal(signal.SIGTERM, sigterm_handler)

def parse(to_parse=None):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training\nlearning rate will decay every 60 epochs')
    parser.add_argument('config_id', type=str, help='config identity str, parsed for lr, lr_decay and minibatch size, looks like: "<lr>_<lr_decay>_<weight-decay>_<minibatch_size>"')
    parser.add_argument('--scratch', '-s', default=os.environ.get('SCRATCH',os.getcwd()), help='place to store data')
    #parser.add_argument('--lr', default=0.1, help='learning rate')
    parser.add_argument('--l1', default=0., type=float, help='l1 regularisation factor')
    parser.add_argument('--l2', default=5e-4, type=float, help='l2 regularisation factor')
    #parser.add_argument('--lr_decay', default=0.01, help='learning rate decay coefficient')
    #parser.add_argument('--minibatch', '-M', default=128, help='minibatch size')
    parser.add_argument('--epochs', '-N', default=180, help='number of epochs to train for')
    parser.add_argument('--gpu', default=0, help='index of gpu to use')
    parser.add_argument('--multi_gpu', action='store_true', help='use all available gpus')
    parser.add_argument('--model', default='resnet50', type=str, help='string to choose model')
    parser.add_argument('--model_multiplier', default=4, type=int, help='multiplier for number of planes in model')
    parser.add_argument('-v', action='store_true', help='verbose with progress bar')
    parser.add_argument('--evaluate', action='store_true', help='run on test set')
    parser.add_argument('--deep_compression', default=1.0, type=float, help='prescribed sparsity to implement with pruning')
    parser.add_argument('--clean', action='store_true', help='Whether to start from clean (WILL DELETE OLD FILES).')
    parser.add_argument('--sgdr', action='store_true', help='use the SGDR learning rate schedule')
    args = parser.parse_args(to_parse)
    return args

# define run_identity for hyperband
def run_identity(argv):
    argv = ["dummy_config"] + argv # have to supply something or hit an error
    args = parse(argv)
    myhost = os.uname()[1].split(".")[0] + "."
    return myhost + format_model_tag(args.model, args.model_multiplier, args.l1, args.deep_compression, args.sgdr)

def get_random_config_id(rng):
    learning_rate = np.exp(rng.uniform(low=np.log(0.01), high=np.log(0.4)))
    lr_decay = 0.2
    #minibatch_size = rng.choice([128,256])
    minibatch_size = 128
    l2 = np.exp(rng.uniform(low=np.log(5e-6), high=np.log(1e-4)))
    config_id = format_settings_str(learning_rate, lr_decay, l2, minibatch_size)
    return config_id

def get_config(config_id):
    config = config_id.split("_")
    return float(config[0]), float(config[1]), float(config[2]), int(config[3])

def format_model_tag(model, model_multiplier, l1, deep_compression, sgdr):
    if 'resnet' in model:
        model_tag = model+".%02d"%model_multiplier+format_l1(l1)
    else:
        model_tag = model+format_l1(l1)
    if deep_compression < 0.99:
        model_tag += '.dc_%05.3f'%deep_compression
    if sgdr:
        model_tag += '.sgdr'
    return model_tag

def main(args):
    if args.v:
        progress_bar = ProgressBar()

    use_cuda = torch.cuda.is_available()
    gpu_index = int(args.gpu) if not args.multi_gpu else None
    best_acc = 0  # best test accuracy

    n_gpus = torch.cuda.device_count()

    # parse out config
    lr, lr_decay, l2, minibatch = get_config(args.config_id)

    trainloader, valloader, testloader = cifar10(args.scratch, minibatch, verbose=args.v)

    # Set where to save and load checkpoints, use model_tag for directory name
    model_tag = format_model_tag(args.model, args.model_multiplier, args.l1, args.deep_compression, args.sgdr)
    if args.v:
        print(model_tag)

    checkpoint_loc = os.path.join(args.scratch, 'checkpoint', model_tag)
    # Set where to append tensorboard logs
    log_loc = os.path.join(args.scratch, 'logs', model_tag)
    if not os.path.isdir(checkpoint_loc):
        os.makedirs(checkpoint_loc)
    if os.path.exists(checkpoint_loc) and args.clean:
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
            return None
    if args.v:
        print("Checkpoint saved to: %s"%checkpoint_loc)

    if not os.path.isdir(log_loc):
        os.makedirs(log_loc)
    if args.v:
        print("Tensorboard logs saved to: %s"%log_loc)

    # define learning rate schedule and make it the default
    # generalised so it would work with any schedule definable with mbatch index
    if args.sgdr:
        schedule = sgdr
        lr_period = 10
    else:
        schedule = standard_schedule
        lr_period = 60

    # complicated way to initialise the Checkpoint object that'll hold our
    # model
    if 'VGG' in model_tag:
        model = VGG('VGG16') # model constructor
    elif 'resnet' in model_tag:
        if '50' in model_tag:
            model = ResNet50(args.model_multiplier, nn.Conv2d, nn.Linear)
        elif 'acdc' in model_tag:
            model = ResNetACDC(args.model_multiplier)
        else:
            raise NotImplementedError("Don't know what model %s should mean."%model_tag)
    elif 'mobilenet' in model_tag:
        model = MobileNet()
    elif 'allconv' == model_tag.split(".")[0]:
        model = AllConv()
    elif 'allconv_acdc' == model_tag.split(".")[0]:
        model = AllConvACDC()
    else:
        raise NotImplementedError("Don't know what model %s should mean."%model_tag)

    # choose model from args
    if args.deep_compression < 0.99:
        from deep_compression import ExactSparsity
        optimizer = ExactSparsity(model.parameters(), args.deep_compression,
                lr=lr, momentum=0.9, weight_decay=l2)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=l2)

    checkpoint = Checkpoint(model, lr, lr_decay, minibatch, schedule,
            checkpoint_loc, log_loc, args.config_id, verbose=args.v,
            multi_gpu=args.multi_gpu, l1_factor=args.l1, l2_factor=l2,
            optimizer=optimizer, lr_period=lr_period)

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
    model_tag = format_model_tag(args.model, args.model_multiplier, args.l1, args.deep_compression, args.sgdr)
    logging_loc = os.path.join(args.scratch, 'checkpoint', model_tag, 'errors.log')
    if not os.path.isdir(os.path.dirname(logging_loc)):
        os.makedirs(os.path.dirname(logging_loc))
    logging.basicConfig(filename=logging_loc,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', 
            datefmt='%m-%d %H:%M', level=logging.DEBUG)
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
