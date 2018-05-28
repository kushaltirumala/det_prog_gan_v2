import argparse
import os
import math
import sys
import pickle
import time
import numpy as np
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bball_data import BBallData
from model import *
from torch.autograd import Variable
from torch import nn
import torch
import torch.utils
import torch.utils.data

from helpers import *
import visdom
from bball_data.utils import unnormalize, plot_sequence, animate_sequence

Tensor = torch.DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

def printlog(line):
    print(line)
    with open(save_path+'log.txt', 'a') as file:
        file.write(line+'\n')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trial', type=int, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--y_dim', type=int, required=True)
parser.add_argument('--h_dim', type=int, required=True)
parser.add_argument('--rnn1_dim', type=int, required=True)
parser.add_argument('--rnn2_dim', type=int, required=True)
parser.add_argument('--rnn4_dim', type=int, required=True)
parser.add_argument('--rnn8_dim', type=int, required=True)
parser.add_argument('--rnn16_dim', type=int, required=True)
parser.add_argument('--n_layers', type=int, required=False, default=2)
parser.add_argument('--seed', type=int, required=False, default=345)
parser.add_argument('--clip', type=int, required=True, help='gradient clipping')
parser.add_argument('--pre_start_lr', type=float, required=True, help='pretrain starting learning rate')
parser.add_argument('--pre_min_lr', type=float, required=True, help='pretrain minimum learning rate')
parser.add_argument('--batch_size', type=int, required=False, default=64)
parser.add_argument('--save_every', type=int, required=False, default=50, help='periodically save model')
parser.add_argument('--pretrain', type=int, required=False, default=50, help='num epochs to use supervised learning to pretrain')
parser.add_argument('--subsample', type=int, required=False, default=1, help='subsample sequeneces')
parser.add_argument('--cuda', action='store_true', default=True, help='use GPU')
parser.add_argument('--cont', action='store_true', default=False, help='continue training a model')
parser.add_argument('--pretrained_discrim', action='store_true', default=False, help='load pretrained discriminator')

parser.add_argument('--discrim_rnn_dim', type=int, required=True)
parser.add_argument('--discrim_layers', type=int, required=True, default=2)
parser.add_argument('--policy_learning_rate', type=float, default=1e-6, help='policy network learning rate for GAN training')
parser.add_argument('--discrim_learning_rate', type=float, default=1e-3, help='discriminator learning rate for GAN training')
parser.add_argument('--max_iter_num', type=int, default=60000, help='maximal number of main iterations (default: 60000)')
parser.add_argument('--log_interval', type=int, default=1, help='interval between training status logs (default: 1)')
parser.add_argument('--draw_interval', type=int, default=50, help='interval between drawing and more detailed information (default: 50)')
parser.add_argument('--pretrain_disc_iter', type=int, default=2000, help="pretrain discriminator iteration (default: 2000)")
parser.add_argument('--save_model_interval', type=int, default=50, help="interval between saving model (default: 50)")

args = parser.parse_args()

if not torch.cuda.is_available():
    args.cuda = False

# model parameters
params = {
    'model' : args.model,
    'y_dim' : args.y_dim,
    'h_dim' : args.h_dim,
    'rnn1_dim' : args.rnn1_dim,
    'rnn2_dim' : args.rnn2_dim,
    'rnn4_dim' : args.rnn4_dim,
    'rnn8_dim' : args.rnn8_dim,
    'rnn16_dim' : args.rnn16_dim,
    'n_layers' : args.n_layers,
    'discrim_rnn_dim' : args.discrim_rnn_dim,
    'discrim_num_layers' : args.discrim_layers,
    'cuda' : args.cuda,
}

# hyperparameters
pretrain_epochs = args.pretrain
clip = args.clip
start_lr = args.pre_start_lr
min_lr = args.pre_min_lr
batch_size = args.batch_size
save_every = args.save_every

# manual seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

# build model
policy_net = PROG_RNN(params)
discrim_net = Discriminator(params).double()
if args.cuda:
    policy_net, discrim_net = policy_net.cuda(), discrim_net.cuda()
params['total_params'] = num_trainable_params(policy_net)
print(params)

# create save path and saving parameters
save_path = 'saved/%03d/' % args.trial
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path+'model/')

# Data
test_data = torch.Tensor(pickle.load(open('bball_data/data/Xte_role.p', 'rb'))).transpose(0, 1)[:, ::args.subsample, :]
train_data = torch.Tensor(pickle.load(open('bball_data/data/Xtr_role.p', 'rb'))).transpose(0, 1)[:, ::args.subsample, :]
if args.subsample == 1:
    test_data = test_data[:, :-1, :]
    train_data = train_data[:, :-1, :]
print(test_data.shape, train_data.shape)

# figures and statistics
if os.path.exists('imgs'):
    shutil.rmtree('imgs')
if not os.path.exists('imgs'):
    os.makedirs('imgs')
vis = visdom.Visdom()
win_pre_policy = None
win_pre_path_length = None
win_pre_out_of_bound = None
win_pre_step_change = None

# continue a previous experiment
if args.cont and args.subsample < 16:
    print("loading model with step size {}...".format(args.subsample*2))
    state_dict = torch.load(save_path+'model/policy_step'+str(args.subsample*2)+'_training.pth')
    policy_net.load_state_dict(state_dict, strict=False)
    
    test_loss = run_epoch(False, args.subsample, policy_net, test_data, clip)
    printlog('Pretrain Test:\t' + str(test_loss))

############################################################################
##################       START SUPERVISED PRETRAIN        ##################
############################################################################

# pretrain
best_test_loss = 0
lr = start_lr
for e in range(pretrain_epochs):
    epoch = e+1
    print("Epoch: {}".format(epoch))

    # draw and stats 
    _, _, _, _, _, _, mod_stats, exp_stats = \
            collect_samples(policy_net, test_data, use_gpu, e, args.subsample, name='pretrain', draw=True)
    update = 'append' if epoch > 1 else None
    win_pre_path_length = vis.line(X = np.array([epoch]), \
        Y = np.column_stack((np.array([exp_stats['ave_length']]), np.array([mod_stats['ave_length']]))), \
        win = win_pre_path_length, update = update, opts=dict(legend=['expert', 'model'], title="average path length"))
    win_pre_out_of_bound = vis.line(X = np.array([epoch]), \
        Y = np.column_stack((np.array([exp_stats['ave_out_of_bound']]), np.array([mod_stats['ave_out_of_bound']]))), \
        win = win_pre_out_of_bound, update = update, opts=dict(legend=['expert', 'model'], title="average out of bound rate"))
    win_pre_step_change = vis.line(X = np.array([epoch]), \
        Y = np.column_stack((np.array([exp_stats['ave_change_step_size']]), np.array([mod_stats['ave_change_step_size']]))), \
        win = win_pre_step_change, update = update, opts=dict(legend=['expert', 'model'], title="average step size change"))

    # control learning rate
    if epoch == pretrain_epochs * 2 // 3:
        lr = min_lr
        print(lr)

    # train
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, policy_net.parameters()),
        lr=lr)

    start_time = time.time()

    train_loss = run_epoch(True, args.subsample, policy_net, train_data, clip, optimizer)
    printlog('Train:\t' + str(train_loss))

    test_loss = run_epoch(False, args.subsample, policy_net, test_data, clip, optimizer)
    printlog('Test:\t' + str(test_loss))

    epoch_time = time.time() - start_time
    printlog('Time:\t {:.3f}'.format(epoch_time))

    total_test_loss = test_loss
    
    update = 'append' if epoch > 1 else None
    win_pre_policy = vis.line(X = np.array([epoch]), Y = np.column_stack((np.array([test_loss]), np.array([train_loss]))), \
        win = win_pre_policy, update = update, opts=dict(legend=['out-of-sample loss', 'in-sample loss'], \
                                                         title="pretrain policy training curve"))

    # best model on test set
    if best_test_loss == 0 or total_test_loss < best_test_loss:    
        best_test_loss = total_test_loss
        filename = save_path+'model/policy_step'+str(args.subsample)+'_state_dict_best_pretrain.pth'
        torch.save(policy_net.state_dict(), filename)
        printlog('Best model at epoch '+str(epoch))

    # periodically save model
    if epoch % save_every == 0:
        filename = save_path+'model/policy_step'+str(args.subsample)+'_state_dict_'+str(epoch)+'.pth'
        torch.save(policy_net.state_dict(), filename)
        printlog('Saved model')
    
printlog('End of Pretrain, Best Test Loss: {:.4f}'.format(best_test_loss))

############################################################################
##################       START ADVERSARIAL TRAINING       ##################
############################################################################

# load the best pretrained policy
policy_state_dict = torch.load(save_path+'model/policy_step'+str(args.subsample)+'_state_dict_best_pretrain.pth')
policy_net.load_state_dict(policy_state_dict)

if args.pretrained_discrim:
    print("loading pretrained discriminator ...")
    discrim_state_dict = torch.load(save_path+'model/discrim_step'+str(args.subsample)+'_pretrained.pth')
    discrim_net.load_state_dict(discrim_state_dict)    
    
# optimizer
optimizer_policy = torch.optim.Adam(
    filter(lambda p: p.requires_grad, policy_net.parameters()),
    lr=args.policy_learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.discrim_learning_rate)
discrim_criterion = nn.BCELoss()
if use_gpu:
    discrim_criterion = discrim_criterion.cuda()

# stats
exp_p = []
win_exp_p = None
mod_p = []
win_mod_p = None
win_path_length = None
win_out_of_bound = None
win_step_change = None
win_nll_loss = None
win_ave_player_dis = None
win_diff_max_min = None
win_ave_angle = None

# select 5 fixed data for testing
fixed_test_data = Variable(test_data[:5].squeeze().transpose(0, 1))
if use_gpu:
    fixed_test_data = fixed_test_data.cuda()

# Pretrain Discriminator
for i in range(args.pretrain_disc_iter):
    exp_states, exp_actions, exp_seq, model_states_var, model_actions_var, model_seq, mod_stats, exp_stats = \
        collect_samples(policy_net, train_data, use_gpu, i, args.subsample, name="pretraining", draw=False)
    model_states = model_states_var.data
    model_actions = model_actions_var.data
    pre_mod_p, pre_exp_p = update_discrim(discrim_net, optimizer_discrim, discrim_criterion, exp_states, \
        exp_actions, model_states, model_actions, i, dis_times=3.0, use_gpu=use_gpu, train=True)

    print(i, 'exp: ', pre_exp_p, 'mod: ', pre_mod_p)

    if pre_mod_p < 0.3:
        break

# Save pretrained model
if args.pretrain_disc_iter > 250:
    torch.save(policy_net.state_dict(), save_path+'model/policy_step'+str(args.subsample)+'_pretrained.pth')
    torch.save(discrim_net.state_dict(), save_path+'model/discrim_step'+str(args.subsample)+'_pretrained.pth')

test_fixed_data(policy_net, fixed_test_data, 'pretrained', 0, args.subsample, 1)

# GAN training
train_discrim = True
for i_iter in range(args.max_iter_num):
    ts0 = time.time()
    print("Collecting Data")
    exp_states, exp_actions, exp_seq, model_states_var, model_actions_var, model_seq, mod_stats, exp_stats = \
        collect_samples(policy_net, train_data, use_gpu, i_iter, args.subsample, draw=False)
    model_states = model_states_var.data
    model_actions = model_actions_var.data    
    
    # draw and stats
    if i_iter % args.draw_interval == 0:
        test_fixed_data(policy_net, fixed_test_data, 'fixed_test', i_iter, args.subsample, 1)
        _, _, _, _, _, _, mod_stats, exp_stats = \
            collect_samples(policy_net, test_data, use_gpu, i_iter, args.subsample, draw=True)
    
        update = 'append' if i_iter > 0 else None
        win_path_length = vis.line(X = np.array([i_iter // args.draw_interval]), \
            Y = np.column_stack((np.array([exp_stats['ave_length']]), np.array([mod_stats['ave_length']]))), \
            win = win_path_length, update = update, opts=dict(legend=['expert', 'model'], title="average path length"))
        win_out_of_bound = vis.line(X = np.array([i_iter // args.draw_interval]), \
            Y = np.column_stack((np.array([exp_stats['ave_out_of_bound']]), np.array([mod_stats['ave_out_of_bound']]))), \
            win = win_out_of_bound, update = update, opts=dict(legend=['expert', 'model'], title="average out of bound rate"))
        win_step_change = vis.line(X = np.array([i_iter // args.draw_interval]), \
            Y = np.column_stack((np.array([exp_stats['ave_change_step_size']]), np.array([mod_stats['ave_change_step_size']]))), \
            win = win_step_change, update = update, opts=dict(legend=['expert', 'model'], title="average step size change"))
        win_ave_player_dis = vis.line(X = np.array([i_iter // args.draw_interval]), \
            Y = np.column_stack((np.array([exp_stats['ave_player_distance']]), np.array([mod_stats['ave_player_distance']]))), \
            win = win_ave_player_dis, update = update, opts=dict(legend=['expert', 'model'], title="average player distance"))
        win_diff_max_min = vis.line(X = np.array([i_iter // args.draw_interval]), \
            Y = np.column_stack((np.array([exp_stats['diff_max_min']]), np.array([mod_stats['diff_max_min']]))), \
            win = win_diff_max_min, update = update, opts=dict(legend=['expert', 'model'], title="average max and min path diff"))
        win_ave_angle = vis.line(X = np.array([i_iter // args.draw_interval]), \
            Y = np.column_stack((np.array([exp_stats['ave_angle']]), np.array([mod_stats['ave_angle']]))), \
            win = win_ave_angle, update = update, opts=dict(legend=['expert', 'model'], title="average rotation angle"))
    ts1 = time.time()

    t0 = time.time()
    # update discriminator
    mod_p_epoch, exp_p_epoch = update_discrim(discrim_net, optimizer_discrim, discrim_criterion, exp_states, exp_actions, \
                                              model_states, model_actions, i_iter, dis_times=3.0, use_gpu=use_gpu, train=train_discrim)
    exp_p.append(exp_p_epoch)
    mod_p.append(mod_p_epoch)
    
    # update policy network
    if i_iter > 3 and mod_p[-1] < 0.8:
        update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, model_states_var, model_actions_var, i_iter, use_gpu)
    t1 = time.time()

    if i_iter % args.log_interval == 0:
        print('{}\tT_sample {:.4f}\tT_update {:.4f}\texp_p {:.3f}\tmod_p {:.3f}'.format(
            i_iter, ts1-ts0, t1-t0, exp_p[-1], mod_p[-1]))
        
        update = 'append'
        if win_exp_p is None:
            update = None
        win_exp_p = vis.line(X = np.array([i_iter]), \
                             Y = np.column_stack((np.array([exp_p[-1]]), np.array([mod_p[-1]]))), \
                             win = win_exp_p, update = update, \
                             opts=dict(legend=['expert_prob', 'model_prob'], title="training curve probs"))

    if args.save_model_interval > 0 and (i_iter) % args.save_model_interval == 0:
        torch.save(policy_net.state_dict(), save_path+'model/policy_step'+str(args.subsample)+'_training.pth')
        torch.save(discrim_net.state_dict(), save_path+'model/discrim_step'+str(args.subsample)+'_training.pth')