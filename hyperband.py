'''
Persistent Hyperband, will get up and keep running if run again after being
killed.
'''

import math
import pickle

import subprocess
import argparse
import os
import sys
import time
import logging

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import torch
n_gpus = torch.cuda.device_count()

from checkpoint import format_settings_str
from utils import format_l1

def get_random_config(rng):
    learning_rate = np.exp(rng.uniform(low=np.log(0.01), high=np.log(0.4)))
    lr_decay = rng.uniform(low=0., high=0.5)
    minibatch_size = 2**rng.randint(low=6, high=9)
    return learning_rate, lr_decay, minibatch_size

def parse():
    parser = argparse.ArgumentParser(description='Hyperband optimiser, runs configs using the `main.py` script.')
    parser.add_argument('--model_multiplier', default=4, type=int, help='multiplier for number of planes in model')
    parser.add_argument('--model', default='resnet50', type=str, help='string referring to model to use')
    parser.add_argument('--max_iter', default=180, type=int, help='maximum number of iterations any model can be run')
    parser.add_argument('--eta', default=5., type=float, help='downsampling rate')
    parser.add_argument('--l1', default=0., type=float, help='l1 coefficient')
    parser.add_argument('--dry', action='store_true', help='dry run')
    return parser.parse_args()

class Hyperband(object):
    """
    Checkpoint handler for hyperband hyperparameter optimisation:

        https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

    Taking defaults from that web page.

    Arguments:
        max_iter: maximum number of iterations per configuration.
        eta: defines downsampling rate (default=3)
    """
    def __init__(self, max_iter=180., eta=5.):
        try:
            self.load_state()
        except EnvironmentError:
            self.eta, self.max_iter = eta, max_iter
            logeta = lambda x: math.log(x)/math.log(eta) # downsampling rate
            self.s_max = int(logeta(max_iter)) # number of unique executions of successive halving
            self.B = (self.s_max + 1)*max_iter # total no. of iterations per successive halving
            self.rng = np.random.RandomState(42) # initialise rng

            # depth versus width controls of successive halving inner loop
            self.s_list = list(range(self.s_max+1))
            
            self.iterations_complete = 0 # track how many we've done
            
    def pickle_fname(self):
        return run_identity()+".hyperband_state.pkl"

    def load_state(self):
        with open(self.pickle_fname(), "rb") as f:
            self.__dict__.update(pickle.load(f))

    def save_state(self):
        with open(self.pickle_fname(), "wb") as f:
            pickle.dump(self.__dict__, f)

    def __iter__(self):
        while len(self.s_list) > 0:
            s = self.s_list[-1]
            # initial number of configurations
            n = int(math.ceil(((self.B/self.max_iter)/(s+1))*self.eta**s))
            # initial number of iterations to run configurations for
            r = self.max_iter*self.eta**(-s)

            if not hasattr(self, 'inner_loop'):
                self.T = [ get_random_config(self.rng) for i in range(n) ]
                self.inner_loop = list(range(s+1))
                self.completed = 0
            while len(self.inner_loop) > 0:
                i = self.inner_loop[0]
                # run each config for r_i iterations
                n_i = n*self.eta**(-i)
                r_i = r*self.eta**(i)
                r_i = r_i

                val_losses = 100.*np.ones(len(self.T))
                
                self.save_state()
                # thanks https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python
                idxd_T = list(enumerate(self.T))
                chunks = [idxd_T[i:i + n_gpus] for i in range(0, len(idxd_T), n_gpus)]
                for j, chunk in enumerate(chunks):
                    idxs, settings = zip(*chunk)
                    before = time.time()
                    results = parallel_call(settings, r_i, self.completed)
                    iter_rate = (time.time() - before)/float(len(chunk)*r_i)
                    self.iterations_complete += len(chunk)*(r_i-self.completed)
                    val_losses[np.array(idxs)] = results
                    yield self.progress(r_i, s, i, (j+1)*len(chunk), len(self.T), np.min(val_losses), self.T[np.argmin(val_losses)], iter_rate)
                # keep track of how many epochs the saved checkpoints have completed already
                self.completed = r_i

                self.T = [self.T[i] for i in np.argsort(val_losses)[0:int(n_i/self.eta)]]
                _ = self.inner_loop.pop(0)
            _ = self.s_list.pop()
            del self.inner_loop
            del self.T

    def progress(self, n_iter, outer_loc, inner_loc, settings_idx, n_settings, best_loss, best_settings, iter_rate):
        """Writes a string defining the current progress of the optimisation."""
        progress_str = "Remaining configs %02d for %03d iter: "%(n_settings, n_iter)
        progress_str += "outer loop %02d/%02d, "%(self.s_max+1-outer_loc, self.s_max+1)
        progress_str += "inner loop %02d/%02d, "%(inner_loc+1, outer_loc+1)
        progress_str += "configs %02d/%02d, "%(settings_idx, n_settings) 
        remaining = self.total_iterations - self.iterations_complete
        progress_str += "time to complete %04.1f hours, "%((iter_rate*remaining)/(60.**2))
        progress_str += "best_loss %05.3f with "%(best_loss)
        progress_str += format_settings_str(*best_settings)
        return progress_str

    def preamble(self, dry=False):
        """Prints full table of what will be run."""
        # yes, the easiest way to run the above algorithm without actually
        # running it was to just copy it
        from collections import defaultdict
        table = defaultdict(dict)
        self.total_iterations = 0
        for s in reversed(self.s_list):
            # initial number of configurations
            n = int(math.ceil(((self.B/self.max_iter)/(s+1))*self.eta**s))
            # initial number of iterations to run configurations for
            r = self.max_iter*self.eta**(-s)
            completed = 0

            T = [ get_random_config(self.rng) for i in range(n) ]
            for i in range(s+1):
                # run each config for r_i iterations
                n_i = n*self.eta**(-i)
                r_i = r*self.eta**(i)
                r_i = r_i

                table[i][s] = [n_i, r_i]

                # thanks https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python
                idxd_T = list(enumerate(T))
                chunks = [idxd_T[i:i + n_gpus] for i in range(0, len(idxd_T), n_gpus)]
                self.total_iterations += (r_i-completed)*len(T)
                completed = r_i

                T = [T[i] for i in np.arange(len(T))[0:int(n_i/self.eta)]]
        # 8 spaces between each
        top_row = "max_iter %03d:   "%int(self.max_iter) + "          ".join(["s=%01d"%s for s in reversed(self.s_list)]) + "\n"
        mtop    = "eta %01d           "%self.eta      + "    ".join(["n_i   r_i" for _ in reversed(self.s_list)]) + "\n"
        btop    = "total: %05d    "%self.total_iterations  + "    ".join(["---------" for _ in reversed(self.s_list)]) + "\n"
        indent  = "                "
        rows = []
        for i in range(self.s_list[-1]+1):
            try:
                row = []
                for s in reversed(self.s_list):
                    n_i, r_i = table[i][s]
                    row.append("%03d   %03d"%(n_i,r_i))

            except KeyError:
                pass
            row = indent + "    ".join(row)
            rows.append(row)
        rows = "\n".join(rows)
        preamble_str = top_row+mtop+btop+rows
        if dry:
            preamble_str += "\nIf each run takes 1 minute then will complete in %.2f hours"%((self.total_iterations/n_gpus)/(60.))
        print(preamble_str)
        logging.info("PREAMBLE:\n"+preamble_str)

def parallel_call(settings_to_run, n_iterations, completed):
    if len(settings_to_run) == 1:
        result = run_settings(settings_to_run[0], n_iterations, None,
                timeout=(n_iterations-completed)*240, multi_gpu=True)
        results = [result]
    else:
        call = lambda settings, gpu_index: run_settings(settings, n_iterations, gpu_index, timeout=n_iterations*240)
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            results = executor.map(call, settings_to_run, range(n_gpus), timeout=n_iterations*500)
            
    return np.array(list(results))
    #results = []
    #for s, i in zip(settings_to_run, range(n_gpus)):
    #    results.append(call(s, i))
    #return np.array(list(results))

if __name__ == '__main__':
    args = parse()
    def run_identity():
       myhost = os.uname()[1].split(".")[0]
       return myhost+".%02d"%args.model_multiplier + format_l1(args.l1)

    def run_settings(settings, n_i, gpu_index, timeout, multi_gpu=False):
        if multi_gpu:
            options = ["--multi_gpu"]
        else:
            options = ["--gpu","%i"%gpu_index]
        options += ["--lr","%f"%settings[0]]
        options += ["--lr_decay","%f"%settings[1]]
        options += ["--minibatch","%i"%settings[2]]
        options += ["--epochs","%i"%n_i]
        options += ["--model_multiplier","%i"%args.model_multiplier]
        options += ["--l1","%f"%args.l1]
        options += ["--model",args.model]
        try:
            command = ['python', 'main.py']+options
            logging.info("RUNNING:  "+ " ".join(command))
            out = subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=timeout)
            loss = float(out.decode("utf-8").split("\n")[-2])
            logging.info("COMPLETE: "+ " ".join(command)+" LOSS: %.3f"%loss)
            return loss
        except KeyboardInterrupt as e:
            raise e
        except subprocess.TimeoutExpired as e:
            error = e.output.decode("utf-8").strip('\n')
            logging.info("FAILED:   "+ " ".join(command) + " ERROR: "+ error)
            return 100.
        except Exception as e:
            if hasattr(e, 'output'):
                error = e.output.decode("utf-8").split('\n')[-2]
                if 'KeyboardInterrupt' in error:
                    error += " epoch took too long to execute"
            else:
                error = str(e).strip()
            logging.info("FAILED:   "+ " ".join(command) + " ERROR: "+ error)
            return 100.0

    # initialise logging
    logging.basicConfig(filename=run_identity()+".hyperband.log", level=logging.DEBUG)

    h = Hyperband(max_iter=args.max_iter, eta=args.eta)
    h.preamble(dry=args.dry)
    if not args.dry:
        for progress in h:
            # update progress bar here
            logging.info("PROGRESS: " + progress)
            sys.stdout.write(progress)
            sys.stdout.write("\r")
            sys.stdout.flush()
        sys.stdout.write("\n")

