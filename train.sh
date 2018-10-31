#!/bin/bash

python3 training.py --agent A2C --environment CartPole-v0 \
--net_type Shared --optimizer Adam --lr 1e-3 --entropy_reg 1e-2 \
--gamma 0.99 \
--cuda false \
--train_steps 5000 --n_environments 5 --rollout_len 5 --normalize_advantage false \
--watch 5