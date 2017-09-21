
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
        max_iter: maximum number of iterations per configuration.
        eta: defines downsampling rate (default=3)
    """
    def __init__(self, get_random_config, max_iter=180, eta=3.):
        self.eta, self.max_iter = eta, max_iter
        self.get_random_config = get_random_config
        self.logeta = lambda x: math.log(x)/math.log(eta) # downsampling rate
        self.s_max = int(self.logeta(max_iter)) # number of unique executions of successive halving
        self.B = (self.s_max + 1)*max_iter # total no. of iterations per successive halving

        self.checkpoint_queue = [] # start with nothing to run

        self.rng = np.random.RandomState(42) # initialise rng

        # initialise configuration generator
        self.configs = self.configuration_generator()

        # initialise queue
        self.pending, self.queue = None, []

    def configuration_generator(self):
        for s in reversed(range(self.s_max+1)):
            # initial number of configurations
            n = int(math.ceil(self.B/self.max_iter/(s+1)*self.eta**s))
            # initial number of iterations to run configurations for
            r = self.max_iter*self.eta**(-s)
            print("Running with %i configurations, max %i iterations."%(n,r))

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
                if len(self.queue) > 0:
                    print("finishing queue:", self.queue)
                    # if we still have something in the queue, don't continue
                    yield "flush"

                T = [T[i] for i in np.argsort(self.val_losses)[0:int(n_i/self.eta)]]

    def populate_queue(self, n):
        # try to add item, destroy checkpoint if it already exists 
        for _ in range(n):
            # make sure pending isn't None before trying to add to queue
            if self.pending is None:
                self.pending = next(self.configs)

            if self.pending is not "flush":
                # add to queue if it's not in there already
                if not any([settings_close(self.pending[1],s) for i,s in self.queue]):
                    self.queue.append((self.pending[0], self.pending[1]))
                    self.pending = None
                else:
                    # if it is in there, don't make queue any bigger
                    print("repeated entry found")
                    break
            else:
                self.pending = None
                break

        print(self.queue)

    def update(self, losses, idxs):
        for i,l in zip(idxs, losses):
            self.val_losses[i] = l

    def get_random_hyperparameter_configution(self):
        raise NotImplementedError()

    def run_then_return_val_loss(self, num_iters, hyperparameters):
        raise NotImplementedError()

def settings_close(settings_a, settings_b):
    total = 0.0
    for a,b in zip(settings_a, settings_b):
        total += abs(a-b)
    return total < 0.01
