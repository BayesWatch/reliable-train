'''
Persistent Hyperband, will get up and keep running if run again after being
killed. So far, will not work perfectly, however.
'''

import math
import pickle

import subprocess
import os

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import torch
n_gpus = torch.cuda.device_count()

def get_random_config(rng):
    learning_rate = np.exp(rng.uniform(low=np.log(0.01), high=np.log(0.2)))
    lr_decay = rng.uniform(low=0., high=0.5)
    minibatch_size = 2**rng.randint(low=4, high=10)
    return learning_rate, lr_decay, minibatch_size

class Hyperband(object):
    """
    Checkpoint handler for hyperband hyperparameter optimisation:

        https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

    Taking defaults from that web page.

    Arguments:
        max_iter: maximum number of iterations per configuration.
        eta: defines downsampling rate (default=3)
    """
    def __init__(self, max_iter=180., eta=3.):
        try:
            self.load_state()
        except:
            self.eta, self.max_iter = eta, max_iter
            logeta = lambda x: math.log(x)/math.log(eta) # downsampling rate
            self.s_max = int(logeta(max_iter)) # number of unique executions of successive halving
            self.B = (self.s_max + 1)*max_iter # total no. of iterations per successive halving
            self.rng = np.random.RandomState(42) # initialise rng

            # depth versus width controls of successive halving inner loop
            self.s_list = list(range(self.s_max+1))

    def load_state(self):
        with open("hyperband_state.pkl", "rb") as f:
            self.__dict__.update(pickle.load(f))

    def save_state(self):
        with open("hyperband_state.pkl", "wb") as f:
            pickle.dump(self.__dict__, f)

    def __iter__(self):
        while len(self.s_list) > 0:
            s = self.s_list[-1]
            # initial number of configurations
            n = int(math.ceil(((self.B/self.max_iter)/(s+1))*self.eta**s))
            # initial number of iterations to run configurations for
            r = self.max_iter*self.eta**(-s)
            print("Running with %i configurations, max %i iterations"%(n,self.max_iter))

            T = [ get_random_config(self.rng) for i in range(n) ]
            if not hasattr(self, 'inner_loop'):
                self.inner_loop = list(range(s+1))
            while len(self.inner_loop) > 0:
                i = self.inner_loop[0]
                # run each config for r_i iterations
                n_i = n*self.eta**(-i)
                r_i = r*self.eta**(i)
                r_i = r_i
                print("    %i configurations left, running for %i iterations"%(len(T), r_i))

                val_losses = 100.*np.ones(len(T))
                self.prescription_idx = 0
                
                self.save_state()
                # thanks https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python
                idxd_T = list(enumerate(T))
                chunks = [idxd_T[i:i + n_gpus] for i in range(0, len(idxd_T), n_gpus)]
                for chunk in chunks:
                    idxs, settings = zip(*chunk)
                    results = parallel_call(settings, r_i)
                    val_losses[np.array(idxs)] = results
                    yield idxs, settings

                T = [T[i] for i in np.argsort(val_losses)[0:int(n_i/self.eta)]]
                _ = self.inner_loop.pop(0)
            _ = self.s_list.pop()
            del self.inner_loop

def run_settings(settings, n_i, gpu_index):
    options = ["--gpu","%i"%gpu_index]
    options += ["--lr","%f"%settings[0]]
    options += ["--lr_decay","%f"%settings[1]]
    options += ["--minibatch","%i"%settings[2]]
    options += ["--epochs","%i"%n_i]
    try:
        #out = subprocess.check_output(['python', 'dummy.py']+options, timeout=360)
        out = subprocess.check_output(['python', 'main.py']+options, timeout=360)
        return float(out.decode("utf-8") .split("\n")[-2])
    except Exception as e:
        print(e)
        return 100.0

def parallel_call(settings_to_run, n_iterations):
    call = lambda settings, gpu_index: run_settings(settings, n_iterations, gpu_index)
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        results = executor.map(call, settings_to_run, range(n_gpus))
        return np.array(list(results))

if __name__ == '__main__':
    h = Hyperband()
    for idxs, settings in h:
        # update progress bar here
        pass
