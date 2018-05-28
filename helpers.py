from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
import os
import struct
from bball_data.utils import unnormalize, plot_sequence
import pickle

use_gpu = torch.cuda.is_available()

# training function used in pretraining
def run_epoch(train, step_size, model, exp_data, clip, optimizer=None, batch_size=64):
    losses = []
    inds = np.random.permutation(exp_data.shape[0])
    
    i = 0
    while i + batch_size <= exp_data.shape[0]:
        ind = torch.from_numpy(inds[i:i+batch_size]).long()
        i += batch_size
        data = exp_data[ind]
    
        if use_gpu:
            data = data.cuda()

        # change (batch, time, x) to (time, batch, x)
        data = Variable(data.squeeze().transpose(0, 1))

        batch_loss = model(data, step_size)

        if train:
            optimizer.zero_grad()
            total_loss = batch_loss
            total_loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
        
        losses.append(batch_loss.data.cpu().numpy()[0])

    return np.mean(losses)

def ones(*shape):
    return torch.ones(*shape).cuda() if use_gpu else torch.ones(*shape)

def zeros(*shape):
    return torch.zeros(*shape).cuda() if use_gpu else torch.zeros(*shape)

# train and pretrain discriminator
def update_discrim(discrim_net, optimizer_discrim, discrim_criterion, exp_states, exp_actions, \
                   states, actions, i_iter, dis_times, use_gpu, train = True):
    if use_gpu:
        exp_states, exp_actions, states, actions = exp_states.cuda(), exp_actions.cuda(), states.cuda(), actions.cuda()

    """update discriminator"""
    g_o_ave = 0.0
    e_o_ave = 0.0
    for _ in range(int(dis_times)):          
        g_o = discrim_net(Variable(states), Variable(actions))
        e_o = discrim_net(Variable(exp_states), Variable(exp_actions))
        
        g_o_ave += g_o.cpu().data.mean()
        e_o_ave += e_o.cpu().data.mean()
        
        if train:
            optimizer_discrim.zero_grad()
            discrim_loss = discrim_criterion(g_o, Variable(zeros((g_o.shape[0], g_o.shape[1], 1)))) + \
                discrim_criterion(e_o, Variable(ones((e_o.shape[0], e_o.shape[1], 1))))
            discrim_loss.backward()
            optimizer_discrim.step()
    
    if dis_times > 0:
        return g_o_ave / dis_times, e_o_ave / dis_times

# train policy network
def update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, \
                  states_var, actions_var, i_iter, use_gpu):
    optimizer_policy.zero_grad()
    g_o = discrim_net(states_var, actions_var)
    policy_loss = discrim_criterion(g_o, Variable(ones((g_o.shape[0], g_o.shape[1], 1))))
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 10)
    optimizer_policy.step()

# sample and draw all 5 level trajectories, used in test_model.py
def test_sample(policy_net, expert_data, use_gpu, i_iter, size=64, name="sampling", draw=False):
    exp_ind = torch.from_numpy(np.random.choice(expert_data.shape[0], size)).long()
    data = expert_data[exp_ind].clone()
    seq_len = data.shape[0]
    if use_gpu:
        data = data.cuda()
    data = Variable(data.squeeze().transpose(0, 1))
    # data: seq_length * batch_size * 10

    samples16 = policy_net.sample16(data[::16], seq_len = 5)
    samples8 = policy_net.sample8(data[::8], seq_len = 8, macro_data = samples16)
    samples4 = policy_net.sample4(data[::4], seq_len = 14, macro_data = samples8)
    samples2 = policy_net.sample2(data[::2], seq_len = 26, macro_data = samples4)
    samples1 = policy_net.sample1(data[::1], seq_len = 50, macro_data = samples2)

    mod_stats = {}
    exp_stats = {}

    if draw:
        _ = draw_data(samples16[:-1].data, name + '_stepsize_16', i_iter)
        _ = draw_data(samples8[:-1].data, name + '_stepsize_8', i_iter)
        _ = draw_data(samples4[:-1].data, name + '_stepsize_4', i_iter)
        _ = draw_data(samples2[:-1].data, name + '_stepsize_2', i_iter)
        mod_stats = draw_data(samples1.data, name, i_iter)
        exp_stats = draw_data(data.data, name + '_expert', i_iter)
    
    return mod_stats, exp_stats    

# sample trajectories used in GAN training
def collect_samples(policy_net, expert_data, use_gpu, i_iter, step_size, size=64, name="sampling", draw=False):
    exp_ind = torch.from_numpy(np.random.choice(expert_data.shape[0], size)).long()
    data = expert_data[exp_ind].clone()
    seq_len = data.shape[0]
    if use_gpu:
        data = data.cuda()
    data = Variable(data.squeeze().transpose(0, 1))
    # data: seq_length * batch_size * 10

    if step_size == 1:
        samples = policy_net.sample1(data)
    elif step_size == 2:
        samples = policy_net.sample2(data)
    elif step_size == 4:
        samples = policy_net.sample4(data)
    elif step_size == 8:
        samples = policy_net.sample8(data)
    elif step_size == 16:
        samples = policy_net.sample16(data)
    
    states = samples[:-1, :, :]
    actions = samples[1:, :, :]
    exp_states = data[:-1, :, :]
    exp_actions = data[1:, :, :]

    mod_stats = {}
    exp_stats = {}

    if draw:
        mod_stats = draw_data(samples.data, name, i_iter)
        exp_stats = draw_data(data.data, name + '_expert', i_iter)
    
    return exp_states.data, exp_actions.data, data.data, states, actions, samples.data, mod_stats, exp_stats

# transfer a 1-d vector into a string
def to_string(x):
    ret = ""
    for i in range(x.shape[0]):
        ret += "{:.3f} ".format(x[i])
    return ret

def ave_player_distance(states):
    # states: numpy (seq_lenth, batch, 10)
    count = 0
    ret = np.zeros(states.shape)
    for i in range(5):
        for j in range(i+1, 5):
            ret[:, :, count] = np.sqrt(np.square(states[:, :, 2 * i] - states[:, :, 2 * j]) + \
                                       np.square(states[:, :, 2 * i + 1] - states[:, :, 2 * j + 1]))
            count += 1
    return ret

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.linalg.norm(v1) == 0.0 or np.linalg.norm(v2) == 0.0:
        return 0.0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.abs(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) - np.pi / 2)

def ave_rotation(actions):
    length = actions.shape[0]
    ret = np.zeros((length-1, actions.shape[1], 5))
    for i in range(length-1):
        for j in range(actions.shape[1]):
            for k in range(5):
                ret[i, j, k] = angle_between(actions[i, j, (2*k):(2*k+2)], actions[i+1, j, (2*k):(2*k+2)])
    return ret

# draw and compute statistics
def draw_data(model_states, name, i_iter, burn_in=0):
    print("Drawing")
    stats = {}
    model_actions = model_states[1:, :, :] - model_states[:-1, :, :]
        
    val_data = model_states.cpu().numpy()
    val_actions = model_actions.cpu().numpy()

    step_size = np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2]))
    change_of_step_size = np.abs(step_size[1:, :, :] - step_size[:-1, :, :])
    stats['ave_change_step_size'] = np.mean(change_of_step_size)
    val_seqlength = np.sum(np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2])), axis = 0)
    stats['ave_length'] = np.mean(val_seqlength)  ## when sum along axis 0, axis 1 becomes axis 0
    stats['ave_out_of_bound'] = np.mean((val_data < -0.51) + (val_data > 0.51))
    
    # more stats added 180425
    stats['ave_player_distance'] = np.mean(ave_player_distance(val_data))
    stats['diff_max_min'] = np.mean(np.max(val_seqlength, axis=1) - np.min(val_seqlength, axis=1))
    stats['ave_angle'] = np.mean(ave_rotation(val_actions))
    
    draw_data = model_states.cpu().numpy()[:, 0, :] 
    draw_data = unnormalize(draw_data)
    colormap = ['b', 'r', 'g', 'm', 'y', 'c']
    plot_sequence(draw_data, macro_goals=None, colormap=colormap[:5], \
                  save_name="imgs/{}_{}_offense".format(name, i_iter), burn_in=burn_in)

    return stats

def test_fixed_data(policy_net, exp_state, name, i_iter, step_size, num_draw=1):
    if step_size == 1:
        samples = policy_net.sample1(exp_state)
    elif step_size == 2:
        samples = policy_net.sample2(exp_state)
    elif step_size == 4:
        samples = policy_net.sample4(exp_state)
    elif step_size == 8:
        samples = policy_net.sample8(exp_state)
    elif step_size == 16:
        samples = policy_net.sample16(exp_state)
    
    draw_data(samples.data, name, i_iter)
    draw_data(exp_state.data, name + '_expert', i_iter)
