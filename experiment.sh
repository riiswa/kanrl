#!/bin/sh
echo "Starting experiments with MLP..."
python3.9 kaqn.py --multirun seed="range(8)" method=MLP width=32
echo "Starting experiments with KAN..."
python3.9 kaqn.py --multirun seed="range(8)"

# TODO can I parallelize runs? 