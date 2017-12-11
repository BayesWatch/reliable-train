#!/bin/bash
set -e

MODEL=$1
GPU=$2

python main.py 0.1_0.1_128 --l1 1e-6 --l2 0. --model $MODEL -v --gpu $GPU
python main.py 0.1_0.1_128 --l1 4.6e-5 --l2 0. --model $MODEL -v --gpu $GPU
python main.py 0.1_0.1_128 --l1 2.1e-5 --l2 0. --model $MODEL -v --gpu $GPU
python main.py 0.1_0.1_128 --l1 1e-4 --l2 0. --model $MODEL -v --gpu $GPU
