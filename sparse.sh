#!/bin/bash

python hyperband.py --max_iter 80 --eta 3 --model_multiplier 1 --l1 0.00005
python hyperband.py --max_iter 80 --eta 3 --model_multiplier 2 --l1 0.00005
python hyperband.py --max_iter 80 --eta 3 --model_multiplier 3 --l1 0.00005
python hyperband.py --max_iter 80 --eta 3 --model_multiplier 4 --l1 0.00005

