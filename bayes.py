"""
Bayesian compression on CIFAR-10
"""
from __future__ import print_function

import torch
from torch import optim

import sys
import os
import argparse
import traceback
import logging

from models import *
from torch.autograd import Variable

import numpy as np

from utils import ProgressBar, sigterm_handler, cleanup
from checkpoint import Checkpoint, format_settings_str, sparsity
from data import cifar10
from seppuku import exit_after
from models.BayesianLayers import LinearGroupNJ, Conv2dGroupNJ, isnt_nan
from models.bayesian_utils import compute_compression_rate

from itertools import combinations

from glob import glob

import signal

# if we receive SIGTERM, we want to log what we were doing when that happened
signal.signal(signal.SIGTERM, sigterm_handler)

def parse(to_parse=None):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training\nlearning rate will decay every 60 epochs')
    parser.add_argument('config_id', type=str, help='config identity str, parsed for lr, lr_decay and minibatch size, looks like: "<lr>_<lr_decay>"')
    parser.add_argument('--scratch', '-s', default=os.environ.get('SCRATCH',os.getcwd()), help='place to store data')
    parser.add_argument('--minibatch', '-M', type=int, default=64, help='minibatch size')
    parser.add_argument('--l2', default=5e-4, type=float, help='l2 regularisation factor')
    parser.add_argument('--epochs', '-N', default=180, help='number of epochs to train for')
    parser.add_argument('--gpu', default=0, help='index of gpu to use')
    parser.add_argument('--multi_gpu', action='store_true', help='use all available gpus')
    parser.add_argument('--model', default='allconv', type=str, help='string to choose model')
    parser.add_argument('--model_multiplier', default=4, type=int, help='multiplier for number of planes in model')
    parser.add_argument('-v', action='store_true', help='verbose with progress bar')
    parser.add_argument('--evaluate', action='store_true', help='run on test set')
    parser.add_argument('--clean', action='store_true', help='Whether to start from clean (WILL DELETE OLD FILES).')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='term to weight of the regularisation term')
    args = parser.parse_args(to_parse)
    return args

# define run_identity for hyperband
def run_identity(argv):
    argv = ["dummy_config"] + argv # have to supply something or hit an error
    args = parse(argv)
    myhost = os.uname()[1].split(".")[0] + "."
    return myhost + format_model_tag(args.model, args.model_multiplier)

def get_random_config_id(rng):
    learning_rate = np.exp(rng.uniform(low=np.log(0.01), high=np.log(0.4)))
    lr_decay = rng.uniform(low=0., high=0.5)
    config_id = format_settings_str(learning_rate, lr_decay)
    return config_id

def get_config(config_id):
    config = config_id.split("_")
    return float(config[0]), float(config[1])

def format_model_tag(model, model_multiplier):
    if 'resnet' in model:
        model_tag = model+".%02d"%model_multiplier
    else:
        model_tag = model
    model_tag += '.bayes'
    return model_tag

def kl_divergence(model):
    """Parses model for layers with a kl_divergence method and aggregates their
    results."""
    KLD = 0.
    for m in model.modules():
        # recursive walk through all modules
        if hasattr(m, 'kl_divergence'):
            KLD += m.kl_divergence()
    return KLD

def clip_variances(model):
    """Parses model for layers with a clip_variances method and runs
    them"""
    for m in model.modules():
        # recursive walk through all modules
        if hasattr(m, 'clip_variances'):
            m.clip_variances()


def main(args):
    if args.v:
        progress_bar = ProgressBar()

    use_cuda = torch.cuda.is_available()
    # using CUDA_VISIBLE_DEVICES, so this should always be 0
    gpu_index = 0
    best_acc = 0  # best test accuracy

    n_gpus = torch.cuda.device_count()

    # parse out config
    lr, lr_decay = get_config(args.config_id)

    # load data
    trainloader, valloader, testloader = cifar10(args.scratch, args.minibatch, verbose=args.v)

    # Set where to save and load checkpoints, use model_tag for directory name
    model_tag = format_model_tag(args.model, args.model_multiplier)

    checkpoint_loc = os.path.join(args.scratch, 'checkpoint', model_tag)
    # Set where to append tensorboard logs
    log_loc = os.path.join(args.scratch, 'logs', model_tag)
    if not os.path.isdir(checkpoint_loc):
        os.makedirs(checkpoint_loc)
    if os.path.exists(checkpoint_loc) and args.clean:
        cleanup(checkpoint_loc, log_loc)
    if args.v:
        print("Checkpoint saved to: %s"%checkpoint_loc)

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
        # VGG16 is the only supported option
        model = VGG('VGG16', Conv2d=Conv2dGroupNJ, Linear=LinearGroupNJ) # model constructor
        #model = Net()
    elif 'resnet' in model_tag:
        if '50' in model_tag:
            model = ResNet50(args.model_multiplier, Conv2dGroupNJ, LinearGroupNJ)
    elif 'allconv' in model_tag:
        model = AllConv(Conv2d=Conv2dGroupNJ)
    else:
        raise NotImplementedError("Don't know what model %s should mean."%model_tag)

    optimizer= optim.Adam(model.parameters(), lr=lr)

    discrimination_loss = nn.functional.cross_entropy

    class ELBO(object):
        """Variationally regularised cross-entropy."""
        def __init__(self, weight=1.0):
            self.model = model
            self.N = float(len(trainloader)*trainloader.batch_size)
            self.weight = weight
        def __call__(self, outputs, targets):
            kl = kl_divergence(model)
            discrimination_error = discrimination_loss(outputs, targets)
            variational_bound = discrimination_error + self.weight*kl.cuda()/self.N
            return variational_bound.cuda()

    checkpoint = Checkpoint(model, lr, lr_decay, args.minibatch, schedule,
            checkpoint_loc, log_loc, optimizer, args.config_id, verbose=args.v,
            multi_gpu=args.multi_gpu, l1_factor=0., CriterionConstructor=ELBO,
            clip_grads_at=0.2)

    #@exit_after(240)
    def train(checkpoint, trainloader):
        checkpoint.init_for_epoch(gpu_index, should_update=True, epoch_size=len(trainloader))

        batch_idx = 0
        train_loss = 0.
        for inputs, targets in trainloader:
            checkpoint.criterion.weight = min(args.kl_weight, float(checkpoint.minibatch_idx)/(len(trainloader)*4))

            batch_idx += 1
            loss = checkpoint.propagate(inputs, targets, batch_idx, should_update=True)
            assert not np.isnan(loss)
            clip_variances(checkpoint.net)
            train_loss += loss

            if args.v:
                progress_str = ''
                progress_str += checkpoint.progress()
                progress_bar(batch_idx, len(trainloader), progress_str)
        train_loss = train_loss/batch_idx
        checkpoint.epoch += 1
        return train_loss
    
    #@exit_after(240)
    def validate(checkpoint, loader, save=False):
        checkpoint.init_for_epoch(gpu_index, should_update=False)

        batch_idx = 0
        train_loss = 0.
        for inputs, targets in loader:
            batch_idx += 1
            loss = checkpoint.propagate(inputs, targets, batch_idx, should_update=False)
            train_loss += loss

            if args.v:
                progress_str = ''
                progress_str += checkpoint.progress()
                progress_bar(batch_idx, len(loader), progress_str)
        if save:
            checkpoint.save_recent()

        layers = [m for m in model.modules() if hasattr(m, 'get_mask')]
        masks  = [l.get_mask() for l in layers]

        # log the current compression rates
        example_idx = checkpoint.minibatch_idx*checkpoint.minibatch_size
        CR_architecture, CR_fast_inference = compute_compression_rate(layers, masks)
        checkpoint.summary_writer.add_scalar('validation/sparsity_compression', CR_architecture, example_idx)
        checkpoint.summary_writer.add_scalar('validation/quant_compression', CR_fast_inference, example_idx)

        return train_loss/batch_idx


    for epoch in range(int(args.epochs) - checkpoint.epoch):
        # train and validate this checkpoint
        if not args.evaluate:
            loss = train(checkpoint, trainloader)
            vloss = validate(checkpoint, valloader, save=True)
            # log loss and sparsity to a file
            sp = sparsity(checkpoint.net if not isinstance(checkpoint.net,
                                    torch.nn.DataParallel) else checkpoint.net.module)
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
        
    print("%f"%checkpoint.most_recent_saved['loss'])

if __name__ == '__main__':
    args = parse()

    # limit GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # initialise logging
    model_tag = format_model_tag(args.model, args.model_multiplier)
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
    except SystemExit:
        raise # should already be logged
    except:
        logging.info("COMMAND DIED MYSTERIOUSLY: %s"%cmdline)

