#!/bin/bash

python hyperband.py --max_iter 80 --eta 3 --model_multiplier 1
python hyperband.py --max_iter 80 --eta 3 --model_multiplier 2
python hyperband.py --max_iter 80 --eta 3 --model_multiplier 3
python hyperband.py --max_iter 80 --eta 3 --model_multiplier 4

