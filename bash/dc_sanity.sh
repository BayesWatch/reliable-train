#!/bin/bash
set -e

MODEL=$1
GPU=$2

python main.py 0.1_0.1_128 --model $MODEL -v --gpu $GPU --deep_compression 0.05
python main.py 0.1_0.1_128 --model $MODEL -v --gpu $GPU --deep_compression 0.01
python main.py 0.1_0.1_128 --model $MODEL -v --gpu $GPU --deep_compression 0.005
python main.py 0.1_0.1_128 --model $MODEL -v --gpu $GPU --deep_compression 0.001
