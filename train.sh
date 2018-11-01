#!/bin/bash

python3 training.py \
--agent PPO --environment CartPole-v1 \
--net_type Shared --optimizer Adam --lr 1e-3 --entropy_reg 1e-1 \
--gamma 0.99 --cuda false --train_steps 5000 \
--n_environments 20 --rollout_len 1 --normalize_advantage true \
--ppo_batch_size 5 --num_ppo_epochs 4 --ppo_eps 0.2 --gae_lambda 0.95 \
--watch 5