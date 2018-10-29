#!/bin/bash

python training.py --agent Reinforce --environment CartPole-v1 \
--env_pool_size 1 \
--net_type Shared --optimizer Adam --lr 1e-3 --entropy_reg 1e-2 \
--gamma 0.99 \
--cuda false \
--train_steps 700