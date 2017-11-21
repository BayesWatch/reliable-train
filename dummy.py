import time
import os

import numpy.random as npr

def get_random_config_id(rng):
    i = rng.randint(0,100) # no config id so sample one
    return str(i)

def get_config(config_id):
    new_rng = npr.RandomState(int(config_id))
    return new_rng.uniform(low=0., high=0.5), new_rng.uniform(low=0., high=0.5)

def run_identity(args):
   myhost = os.uname()[1].split(".")[0]+".".join(args)
   return myhost+".dummy"

if __name__ == '__main__':
    # return random scores and occasionally throw errors
    time.sleep(0.1)
    if npr.rand() < 0.95:
        print(npr.rand())
    else:
        raise ValueError("Dummy error, supposed to appear in logs")

