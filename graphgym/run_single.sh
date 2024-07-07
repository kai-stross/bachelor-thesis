#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
python main.py --cfg configs/pyg/gcn_base.yaml --repeat 1
python main.py --cfg configs/pyg/sage_base.yaml --repeat 1