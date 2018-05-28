#!/bin/bash

python train.py \
--trial 101 \
--model PROG_RNN \
--y_dim 10 \
--h_dim 200 \
--rnn1_dim 300 \
--rnn2_dim 250 \
--rnn4_dim 200 \
--rnn8_dim 150 \
--rnn16_dim 100 \
--n_layers 2 \
--clip 10 \
--pre_start_lr 1e-3 \
--pre_min_lr 1e-4 \
--batch_size 64 \
--pretrain 30 \
--subsample 8 \
--discrim_rnn_dim 128 \
--discrim_layers 1 \
--policy_learning_rate 1e-5 \
--discrim_learning_rate 1e-3 \
--pretrain_disc_iter 2000 \
--max_iter_num 60000 \
--cuda \
--cont
