#!/bin/bash

python3 training.py --agent Reinforce --environment Breakout-v0 \
--env_pool_size 1 \
--net_type Shared --optimizer Adam --lr 1e-3 --entropy_reg 1e-2 \
--gamma 0.999 \
--cuda false \
--train_steps 100000 \
--watch 5