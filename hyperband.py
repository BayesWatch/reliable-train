'''
Persistent Hyperband, will get up and keep running if run again after being
killed.
'''

import math
import pickle
import imp
import subprocess
import argparse
import os
import sys
import time
import logging

import numpy as np

# framework agnostic way to count gpus
from glob import glob
n_gpus = len(glob("/proc/driver/nvidia/gpus/*"))

parser = argparse.ArgumentParser(description='Hyperband optimiser, runs configurations on the supplied --script.') 
parser.add_argument('script', type=str, help='script implementing experiment to run')
parser.add_argument('--max_iter', default=180, type=int, help='maximum number of iterations any model can be run')
parser.add_argument('--eta', default=5., type=float, help='downsampling rate')
parser.add_argument('--dry', action='store_true', help='dry run')
parser.add_argument('--clean', action='store_true', help='deletes all logs and state files, then runs, **use with caution**')

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
        return run_identity+".hyperband_state.pkl"

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
                self.T = [ get_random_config_id(self.rng) for i in range(n) ]
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
                    yield self.progress(s, i, (j+1)*len(chunk), len(self.T), np.min(val_losses), self.T[np.argmin(val_losses)], iter_rate)
                # keep track of how many epochs the saved checkpoints have completed already
                self.completed = r_i

                self.T = [self.T[i] for i in np.argsort(val_losses)[0:int(n_i/self.eta)]]
                _ = self.inner_loop.pop(0)
            _ = self.s_list.pop()
            del self.inner_loop
            del self.T

    def progress(self, outer_loc, inner_loc, settings_idx, n_settings, best_loss, best_settings, iter_rate):
        """Writes a string defining the current progress of the optimisation."""
        progress_str = "Completed: "
        progress_str += "outer loop %02d/%02d, "%(self.s_max+1-outer_loc, self.s_max+1)
        progress_str += "inner loop %02d/%02d, "%(inner_loc+1, outer_loc+1)
        progress_str += "configs %02d/%02d, "%(settings_idx, n_settings) 
        remaining = self.total_iterations - self.iterations_complete
        progress_str += "time to complete %04.1f hours, "%((iter_rate*remaining)/(60.**2))
        progress_str += "best_loss %05.3f with "%(best_loss)
        progress_str += best_settings
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

            T = [ get_random_config_id(self.rng) for i in range(n) ]
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
    experiments = []
    if len(settings_to_run) == 1:
        e = Experiment(settings_to_run[0], n_iterations, None,
                timeout=(n_iterations-completed)*240, multi_gpu=True)
        experiments.append(e)
    else:
        for gpu_idx, s in enumerate(settings_to_run):
            e = Experiment(s, n_iterations, gpu_idx,
                timeout=(n_iterations-completed)*240)
            experiments.append(e)
    # poll all processes until they're all done
    while all([e.process.poll() is None for e in experiments]):
        time.sleep(0.5)
    # gather results from each process
    results = [e.get_result() for e in experiments]
    return np.array(results)

if __name__ == '__main__':
    args, unknown_args = parser.parse_known_args()
    # pull required functions out of supplied scripts
    script = imp.new_module('script')
    with open(args.script, 'r') as f:
        exec(f.read(), script.__dict__)
    # identity of the run depends on what args the script gets passed
    run_identity = script.run_identity(unknown_args)
    get_random_config_id = script.get_random_config_id

    class Experiment(object):
        def __init__(self, settings, n_i, gpu_index, timeout, multi_gpu=False):
            self.options = [settings]
            if multi_gpu:
                self.options += ["--multi_gpu"]
            else:
                self.options += ["--gpu","%i"%gpu_index]
            self.options += ["--epochs","%i"%n_i]
            self.options += unknown_args # pass any unknown args to the script
            self.command = ['python', args.script]+self.options
            logging.info("RUNNING:  "+ " ".join(self.command))
            self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def get_result(self):
            out, error = self.process.communicate()
            if self.process.returncode == 1:
                logging.info("FAILED:   "+ " ".join(self.command) + " ERROR: "+ error.decode("utf-8"))
                return 100.
            else:

                loss = float(out.decode("utf-8").split("\n")[-2])
                logging.info("COMPLETE: "+ " ".join(self.command)+" RESULT: %.3f"%loss)
                return loss

    # clean up state before starting, if specified
    if args.clean and os.path.exists(run_identity+".hyperband.log"):
        os.remove(run_identity+".hyperband.log")
        os.remove(run_identity+".hyperband_state.pkl")

    logging.basicConfig(filename=run_identity+".hyperband.log",
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', 
            datefmt='%m-%d %H:%M', level=logging.DEBUG)

    h = Hyperband(max_iter=args.max_iter, eta=args.eta)
    h.preamble(dry=args.dry)
    if not args.dry:
        for progress in h:
            # simple progress bar
            logging.info("PROGRESS: " + progress)
            sys.stdout.write(progress)
            sys.stdout.write("\r")
            sys.stdout.flush()
        sys.stdout.write("\n")


