import time
import os
import argparse

import numpy.random as npr

parser = argparse.ArgumentParser(description='Dummy experiment')
parser.add_argument('config_id', type=str, help='randomly sampled config identity')
parser.add_argument('--epochs', '-N', type=int, default=180, help='number of epochs to train for')
parser.add_argument('--gpu', default=0, type=int, help='index of gpu to use')
parser.add_argument('--multi_gpu', action='store_true', help='use all available gpus')

def get_random_config_id(rng):
    i = rng.randint(0,100) # no config id so sample one
    return str(i)

def get_config(config_id):
    new_rng = npr.RandomState(int(config_id))
    return new_rng.uniform(low=0., high=0.5), new_rng.uniform(low=0., high=0.5)

def run_identity(args):
   myhost = os.uname()[1].split(".")[0]+".".join(args)
   return myhost+".dummy"

def save(config_id, result):
    if not os.path.exists("dummy"):
        os.mkdir("dummy")
    with open(os.path.join("dummy",config_id), 'w') as f:
        f.write("%f"%result)

def load(config_id):
    with open(os.path.join("dummy",config_id), 'r') as f:
        return float(f.read())

if __name__ == '__main__':
    # return random scores and occasionally throw errors
    time.sleep(0.1)
    args = parser.parse_args()
    config = get_config(args.config_id) # not using it here
    print("I would use gpu %i if I could"%args.gpu)
    if args.multi_gpu:
        print("I can use all available GPUs if I want!")
    if npr.rand() < 0.95:
        try:
            result = load(args.config_id)/args.epochs
        except FileNotFoundError:
            result = npr.rand()/args.epochs
        save(args.config_id, result)
        print(result)
    else:
        raise ValueError("Dummy error, supposed to appear in logs")

