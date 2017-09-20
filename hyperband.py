
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
        self.get_random_config = get_random_config
        self.get_checkpoint = get_checkpoint
        self.logeta = lambda x: math.log(x)/math.log(eta) # downsampling rate
        self.s_max = int(self.logeta(max_iter)) # number of unique executions of successive halving
        self.B = (self.s_max + 1)*max_iter # total no. of iterations per successive halving

        self.checkpoint_queue = [] # start with nothing to run

        self.rng = np.random.RandomState(42) # initialise rng

        # initialise configuration generator
        self.configs = self.configuration_generator()

        # initialise queue
        self.queue = []

    def configuration_generator(self):
        for s in reversed(range(self.s_max+1)):
            # initial number of configurations
            n = int(math.ceil(self.B/self.max_iter/(s+1)*self.eta**s))
            # initial number of iterations to run configurations for
            r = self.max_iter*self.eta**(-s)

            T = [ self.get_random_config(self.rng) for i in range(n) ]
            for i in range(s+1):
                # run each config for r_i iterations
                n_i = n*self.eta**(-i)
                r_i = r*self.eta**(i)

                self.val_losses = 100.*np.ones(len(T))
                self.prescription_idx = 0
                
                for _ in range(int(r_i)):
                    for t_idx, t in enumerate(T):
                        yield t_idx, t

                T = [T[i] for i in np.argsort(self.val_losses)[0:int(n_i/self.eta)]]

    def populate_queue(self, n):
        # try to add item, destroy checkpoint if it already exists 
        while len(self.queue) < n:
            idx, settings = next(self.configs)
            checkpoint = self.get_checkpoint(*settings)
            if checkpoint.setting_str not in [c.setting_str for c in
                                              self.queue]:
                self.queue.append((idx, checkpoint))
            else:
                del checkpoint

    def update_losses(self, losses, idxs):
        for i in range(len(self.val_losses)):
            self.val_losses[i] = self.prescriptions[i]['checkpoint'].best_saved.get('loss', 100.0)

    

    def get_random_hyperparameter_configution(self):
        raise NotImplementedError()

    def run_then_return_val_loss(self, num_iters, hyperparameters):
        raise NotImplementedError()

