#!/bin/sh
echo "Starting experiments with MLP..."
python run_experiment.py --multirun seed="range(32)" method=MLP width=32
echo "Starting experiments with KAN..."
python run_experiment.py --multirun seed="range(32)"