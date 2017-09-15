
import math

import numpy as np

class Hyperband(object):
    """
    Checkpoint handler for hyperband hyperparameter optimisation:

        https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

    Taking defaults from that web page.

    Arguments:
        get_random_config: function that returns random hyperparameter
            configurations.
        get_checkpoint: function that takes a single config and returns a 
            checkpoint.
        max_iter: maximum number of iterations per configuration.
        eta: defines downsampling rate (default=3)
    """
    def __init__(self, get_random_config, get_checkpoint, max_iter=180, eta=3.):
        self.eta, self.max_iter = eta, max_iter
        self.logeta = lambda x: math.log(x)/math.log(eta) # downsampling rate
        self.s_max = int(self.logeta(max_iter)) # number of unique executions of successive halving
        self.B = (self.s_max + 1)*max_iter # total no. of iterations per successive halving

        self.checkpoint_queue = [] # start with nothing to run

        self.rng = np.random.RandomState(42) # initialise rng

        # initialise configuration generator
        self.config_generator = self.configuration_generator()

        self.prescriptions = [(0,0)]

    def configuration_generator(self):
        for s in reversed(range(self.s_max+1)):
            # initial number of configurations
            n = int(ceil(self.B/self.max_iter/(s+1)*self.eta**s))
            # initial number of iterations to run configurations for
            r = self.max_iter*self.eta**(-s)

            T = [ self.get_random_config(self.rng) for i in range(n) ]
            for i in range(s+1):
                # run each config for r_i iterations
                n_i = n*self.eta**(-i)
                r_i = r*self.eta**(i)

                self.val_losses = np.zeros(len(T))
                
                prescriptions = [(get_checkpoint(*t), r_i) for t in T]
                yield prescriptions

                T = [T[i] for i in np.argsort(self.val_losses)[0:int(n_i/eta)]]

    def get_next_checkpoint(self):
        while not all([pc[1]>0 for pc in prescriptions]):
            # return each checkpoint until they run out of prescribed epochs
            # then fill self.val_losses with the current losses and iterate the config generator
            prescription = prescriptions.pop(0)
            prescriptions.append(prescription)

            if prescription[1] > 0:
                prescription[1] += -1
                return prescription[0]

        # we must have run out of prescribed epochs, get some more
        prescriptions = next(self.configuration_generator)

    def get_random_hyperparameter_configution(self):
        raise NotImplementedError()

    def run_then_return_val_loss(self, num_iters, hyperparameters):
        raise NotImplementedError()

