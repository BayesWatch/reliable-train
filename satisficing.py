'''Train on CIFAR10 persistently (script can stop and start as much as you
want, *it'll keep trying*).'''
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

from utils import ProgressBar, format_l2
from checkpoint import Checkpoint, format_settings_str, sparsity
from data import cifar10
from seppuku import exit_after

from itertools import combinations

from glob import glob

from sklearn.ensemble import ExtraTreesRegressor

def parse(to_parse=None):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training\nlearning rate will decay every 60 epochs')
    parser.add_argument('config_id', type=str, help='config identity str, parsed for lr, lr_decay and minibatch size, looks like: "<lr>_<lr_decay>_<minibatch_size>"')
    parser.add_argument('--scratch', '-s', default=os.environ.get('SCRATCH',os.getcwd()), help='place to store data')
    parser.add_argument('--l1', default=0., type=float, help='l1 regularisation factor')
    parser.add_argument('--minibatch', '-M', type=int, default=128, help='minibatch size')
    parser.add_argument('--l2', default=5e-4, type=float, help='l2 regularisation factor')
    parser.add_argument('--epochs', '-N', default=180, help='number of epochs to train for')
    parser.add_argument('--gpu', default=0, help='index of gpu to use')
    parser.add_argument('--multi_gpu', action='store_true', help='use all available gpus')
    parser.add_argument('--model', default='resnet50', type=str, help='string to choose model')
    parser.add_argument('--model_multiplier', default=4, type=int, help='multiplier for number of planes in model')
    parser.add_argument('-v', action='store_true', help='verbose with progress bar')
    parser.add_argument('--evaluate', action='store_true', help='run on test set')
    parser.add_argument('--deep_compression', action='store_true', help='use deep compression to sparsify')
    parser.add_argument('--clean', action='store_true', help='Whether to start from clean (WILL DELETE OLD FILES).')
    args = parser.parse_args(to_parse)
    return args

# define run_identity for hyperband
def run_identity(argv):
    argv = ["dummy_config"] + argv # have to supply something or hit an error
    args = parse(argv)
    myhost = os.uname()[1].split(".")[0] + "."
    return myhost + format_model_tag(args.model, args.model_multiplier, args.l1, args.l2, args.deep_compression) 

def get_random_config_id(rng):
    learning_rate = np.exp(rng.uniform(low=np.log(0.01), high=np.log(0.4)))
    lr_decay = rng.uniform(low=0., high=0.5)
    l1 = np.exp(rng.uniform(low=np.log(1e-7), high=np.log(1e-4)))
    config_id = format_settings_str(learning_rate, lr_decay, l1)
    return config_id

def get_config(config_id):
    config = config_id.split("_")
    return float(config[0]), float(config[1]), float(config[2])

def format_model_tag(model, model_multiplier, l1, l2, deep_compression):
    if 'resnet' in model:
        model_tag = model+".%02d"%model_multiplier+format_l2(l2)
    else:
        model_tag = model
    if deep_compression:
        model_tag += '.dc'
    model_tag += '.satisficing'
    return model_tag

def load_satisficing_history():
    # opens all files in satisficing directory
    history_files = glob("satisficing/*")
    X, Y = [], []
    for h in history_files:
        with open(h, 'r') as f:
            s = f.read()
        lines = s.split('\n')[:-1]
        h = os.path.split(h)[1]
        h = [float(x) for x in h.split('_')]
        for l in lines:
            l = [float(x) for x in l.split(',')]
            # lr, lr_decay, minibatch_size, epoch_idx
            X.append((h[0], h[1], h[2], l[0]))
            Y.append((l[1], l[2]))
    return np.array(X), np.array(Y)

def objective(loss, sparsity):
    """
    Implied objective of sparsity methods in general is to have a negligible
    drop in accuracy, while maximising sparsity. We take that to mean 1% 
    drop in accuracy from the non-sparse resnet50 score.
    """
    if loss > 0.0766: # -log(0.9262)
        return 10.*loss # strictly larger than sparsity we care about
    else:
        return sparsity # hopefully < 0.2 at most

def estimator_samples(rf, X):
    # Check data
    X = rf._validate_X_predict(X)

    # Assign chunk of trees to jobs
    n_jobs, _, _ = _partition_estimators(rf.n_estimators, rf.n_jobs)
    
    # Accumulate all outputs in this list
    y_hat = []
    
    # Parallel loop
    lock = threading.Lock()
    Parallel(n_jobs=n_jobs, verbose=rf.verbose, backend="threading")(
        delayed(accumulate_prediction)(e.predict, X, [y_hat], lock)
        for e in rf.estimators_)
    y_hat = np.array(y_hat)
    return y_hat

def model_value(epoch_idx,config_id,X,Y):
    """
    Take the history (X,Y) and return the expected value of the objective when
    this model finishes training. 
    """
    # fit extra trees forest
    rf = ExtraTreesRegressor(n_estimators=100)
    rf.fit(X,Y)
    # predict based on a new datapoint
    if np.max(X[:,3]) < 179:
        # if we've never seen a final epoch before, just go one epoch into the future
        final_epoch = epoch_idx + 1.
    else:
        final_epoch = 179.
    x = np.array([float(x) for x in config_id.split('_')]+[final_epoch]).reshape(1,-1)
    pl, psp = rf.predict(x).ravel() # predicted loss and sparsity
    predicted_objective = objective(pl, psp)
    # take all trees in the forest as samples from the model
    return predicted_objective

def main(args):
    if args.v:
        progress_bar = ProgressBar()

    use_cuda = torch.cuda.is_available()
    gpu_index = int(args.gpu) if not args.multi_gpu else None
    best_acc = 0  # best test accuracy

    n_gpus = torch.cuda.device_count()

    # parse out config
    lr, lr_decay, l1 = get_config(args.config_id)

    # load data
    trainloader, valloader, testloader = cifar10(args.scratch, args.minibatch, verbose=args.v)

    # Set where to save and load checkpoints, use model_tag for directory name
    model_tag = format_model_tag(args.model, args.model_multiplier, args.l1, args.l2, args.deep_compression)

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

    # choose model from args
    if args.deep_compression:
        from deep_compression import MaskedSGD
        Optimizer = MaskedSGD
    else:
        Optimizer = optim.SGD

    checkpoint = Checkpoint(model, lr, lr_decay, args.minibatch,
            schedule, checkpoint_loc, log_loc, verbose=args.v,
            multi_gpu=args.multi_gpu, l1_factor=args.l1, l2_factor=args.l2,
            Optimizer=Optimizer)

    @exit_after(240)
    def train(checkpoint, trainloader):
        checkpoint.init_for_epoch(gpu_index, should_update=True, epoch_size=len(trainloader))

        batch_idx = 0
        train_loss = 0.
        for inputs, targets in trainloader:
            batch_idx += 1
            loss = checkpoint.propagate(inputs, targets, batch_idx, should_update=True)
            train_loss += loss

            if args.v:
                progress_str = ''
                progress_str += checkpoint.progress()
                progress_bar(batch_idx, len(trainloader), progress_str)
        train_loss = train_loss/batch_idx
        checkpoint.epoch += 1
        return train_loss
    
    @exit_after(240)
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
        return train_loss/batch_idx

    for epoch in range(int(args.epochs) - checkpoint.epoch):
        # train and validate this checkpoint
        if not args.evaluate:
            loss = train(checkpoint, trainloader)
            vloss = validate(checkpoint, valloader, save=True)
            # log loss and sparsity to a file
            sp = sparsity(checkpoint.net if not isinstance(checkpoint.net,
                                    torch.nn.DataParallel) else checkpoint.net.module)
            if not os.path.exists("satisficing"):
                os.mkdir("satisficing")
            with open(os.path.join("satisficing",args.config_id), "a") as f:
                f.write("%i, %.4f, %.4f\n"%(epoch,vloss,sp))
            # save the predicted objective and objective to the tensorboard logs
            o = objective(vloss, sp)
            X,Y = load_satisficing_history()
            po = model_value(epoch,args.config_id,X,Y)
            example_idx = checkpoint.minibatch_idx*checkpoint.minibatch_size
            checkpoint.summary_writer.add_scalar('validation/objective', o, example_idx)
            checkpoint.summary_writer.add_scalar('validation/predicted_objective', po, example_idx)
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
    model_tag = format_model_tag(args.model, args.model_multiplier, args.l1, args.l2, args.deep_compression)
    if args.deep_compression:
        model_tag += '.dc'
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
