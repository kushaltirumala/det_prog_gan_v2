#!/bin/bash

python test_model.py \
--trial 101 \
--max_iter_num 60000 \
--model PROG_RNN \
--y_dim 10 \
--h_dim 200 \
--rnn1_dim 300 \
--rnn2_dim 250 \
--rnn4_dim 200 \
--rnn8_dim 150 \
--rnn16_dim 100 \
--n_layers 2  \
--subsample 1
