
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

        # get our first prescriptions
        self.prescriptions = next(self.configs)

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

                self.val_losses = np.zeros(len(T))
                self.prescription_idx = 0
                
                new_prescriptions = tuple([{'checkpoint':self.get_checkpoint(*t), 'iterations':int(r_i)} for t in T])
                prescriptions, T_new = [], []
                setting_strs = []
                for p_idx, p in enumerate(prescriptions):
                    setting_str = p['checkpoint'].setting_str
                    if setting_str in setting_strs:
                        # we can't have setting collisions
                        print("Found setting collision: %s"%setting_str)
                    else:
                        prescriptions.append(p)
                        T_new.append(T[p_idx])
                    setting_strs.append(setting_str)
                T = T_new
                yield prescriptions

                T = [T[i] for i in np.argsort(self.val_losses)[0:int(n_i/self.eta)]]

    def get_next_checkpoint(self):
        while True:
            while any([pc[1]>0 for pc in self.prescriptions]):
                # return each checkpoint until they run out of prescribed epochs
                # then fill self.val_losses with the current losses and iterate the config generator
                if self.prescription_idx < (len(self.prescriptions)-1):
                    self.prescription_idx += 1
                else:
                    self.prescription_idx = 0
                try:
                    prescription = self.prescriptions[self.prescription_idx]
                except:
                    import pdb
                    pdb.set_trace()

                if prescription['iterations'] > 0:
                    prescription['iterations'] += -1
                    return prescription['checkpoint']

            # we must have run out of prescribed epochs, get some more
            print("Getting next prescribed epochs...")
            for i in range(len(self.val_losses)):
                self.val_losses[i] = self.prescriptions[i][0].best_saved.get('loss', 100.0)
            try:
                self.prescriptions = next(self.configs)
            except StopIteration as e:
                print("Finished finite horizon loop, starting again...")
                # initialise configuration generator
                self.configs = self.configuration_generator()

    def get_random_hyperparameter_configution(self):
        raise NotImplementedError()

    def run_then_return_val_loss(self, num_iters, hyperparameters):
        raise NotImplementedError()

