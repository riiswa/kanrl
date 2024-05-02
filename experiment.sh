#!/bin/sh
echo "Starting experiments with MLP..."
python kaqn.py --multirun seed="range(32)" method=MLP width=32
echo "Starting experiments with KAN..."
python kaqn.py --multirun seed="range(32)"