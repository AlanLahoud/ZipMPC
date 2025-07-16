#Copyright (c) 2025 ETH Zurich, Institute for Dynamics Systems and Control , 
#and Örebro University (AASS), Rahel Rickenbach, Alan Lahoud, Erik Schaffernicht, 
#Melanie N. Zeilinger Johannes A. Stork. No rights reserved.

import torch
import torch.nn as nn
import numpy as np
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor

import utils_new
import torch.autograd.functional as F

from mpc.track.src import simple_track_generator, track_functions
from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx

from torch.optim.lr_scheduler import StepLR

import time

import argparse

import sys
from sys import exit


def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters for the program.')

    parser.add_argument('--mpc_T', type=int, default=9)
    parser.add_argument('--mpc_H', type=int, default=20)
    parser.add_argument('--n_Q', type=int, default=3)
    parser.add_argument('--l_r', type=float, default=0.10)
    parser.add_argument('--v_max', type=float, default=1.8)
    parser.add_argument('--delta_max', type=float, default=0.40)
    parser.add_argument('--p_sigma_manual', type=float, default=3.0)

    return parser.parse_args()


# Parsing arguments
args = parse_arguments()

mpc_T = args.mpc_T
mpc_H = args.mpc_H
n_Q = args.n_Q

mpc_L = 5
#n_Q = mpc_T

l_r = args.l_r
v_max = args.v_max
delta_max = args.delta_max

p_sigma_manual = args.p_sigma_manual

load_model = False


seed_n = 0
torch.manual_seed(seed_n)
np.random.seed(seed_n)

k_curve = 25.
dt = 0.03

l_f = l_r

assert mpc_T%n_Q==0

a_max = 1.5

track_density = 300
track_width = 0.5
max_track_width_perc_casadi = 0.68

bound_d_casadi = 0.5*max_track_width_perc_casadi*track_width

t_track = 0.3
init_track = [0,0,0]

max_p = 100

str_model = f'im_{mpc_T}_{mpc_H}_{n_Q}_{l_r}_{delta_max}_{v_max}_{p_sigma_manual}'

params = torch.tensor([l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, mpc_T])
params_H = torch.tensor([l_r, l_f, track_width, dt, k_curve, v_max, delta_max, a_max, mpc_H])

gen = simple_track_generator.trackGenerator(track_density,track_width)
track_name = 'DEMO_TRACK'

track_function = {
    'DEMO_TRACK'    : track_functions.demo_track,
    'HARD_TRACK'    : track_functions.hard_track,
    'LONG_TRACK'    : track_functions.long_track,
    'LUCERNE_TRACK' : track_functions.lucerne_track,
    'BERN_TRACK'    : track_functions.bern_track,
    'INFINITY_TRACK': track_functions.infinity_track,
    'SNAIL_TRACK'   : track_functions.snail_track
}.get(track_name, track_functions.demo_track)

track_function(gen, t_track, init_track)
gen.populatePointsAndArcLength()
gen.centerTrack()
track_coord = torch.from_numpy(np.vstack(
    [gen.xCoords,
     gen.yCoords,
     gen.arcLength,
     gen.tangentAngle,
     gen.curvature]))

true_dx = utils_new.FrenetKinBicycleDx(track_coord, params, 'cpu')


x0 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
u0 = torch.tensor([0.0, 0.0])


dx=4
du=2

# BS = 40
# u_lower = torch.tensor([-a_max, -delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
# u_upper = torch.tensor([a_max, delta_max]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(dev)
# u_init= torch.tensor([0.1, 0.0]).unsqueeze(0).unsqueeze(0).repeat(mpc_T, BS, 1)#.to(device)
# eps=0.00001
# lqr_iter = 18
#
# grad_method = GradMethods.AUTO_DIFF
#
# #model = utils_new.SimpleNN(mpc_H, n_Q, 5, max_p)
# model = utils_new.TCN(mpc_H, n_Q, 2, max_p)
#
# if load_model==True:
#     try:
#         model.load_state_dict(torch.load(f'./saved_models/model_{str_model}_0.pkl'))
#         print('Model loaded')
#     except:
#         print('No model found to load')
#
# #opt = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
# #opt = torch.optim.RMSprop(model.parameters(), lr=0.0001)
# opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

gpr = GaussianProcessRegressor(random_state=0)

control = utils_new.CasadiControl(track_coord, params)
Q_manual = np.repeat(np.expand_dims(np.array([0.0, 3., 0.5, 0.1, 0, 0.1, 1, 1, 0.1, 0.5]), 0), mpc_T, 0)
p_manual = np.repeat(np.expand_dims(np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), mpc_T, 0)

control_H = utils_new.CasadiControl(track_coord, params_H)
Q_manual_H = np.repeat(np.expand_dims(np.array([0.0, 3., 0.5, 0.1, 0, 0.1, 1, 1, 0.1, 0.5]), 0), mpc_H, 0)
p_manual_H = np.repeat(np.expand_dims(np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0]), 0), mpc_H, 0)

idx_to_casadi = [5,1,2,3,8,9]

# here we should decide how many parameters we would like to learn
learned_param = 2
idx_to_learned_param = [8,9]
bo_bound = 2
bo_step = 10
array1 = np.linspace(-bo_bound, bo_bound, bo_step).tolist()
array2 = np.linspace(-bo_bound, bo_bound, bo_step).tolist()
array3 = np.linspace(-bo_bound, bo_bound, bo_step).tolist()
bo_grid = np.array(list(itertools.product(array1,array2)))
#bo_grid = list(itertools.product([(np.linspace(-bo_bound, bo_bound, bo_step)).tolist() for _ in range(learned_param)]))
grid_shape = np.shape(bo_grid)

# now we sample from this grid without sampling any location twice
samples = np.zeros(0)
warm_start = 5

for i in range(warm_start):
    sample = np.random.randint(0,grid_shape[0],1)
    while np.isin(sample,samples):
        sample = np.random.randint(0,grid_shape[0],1)

    samples = np.append(samples,sample,axis=0)

samples = samples.astype(int)

#run lap with new samples
beta = 100.0
bo_iter = 100
p_bo_base = np.array([0, 0, 0, 0, 0, -p_sigma_manual, 0, 0, 0, 0])
p_bo_add = np.zeros(10)
Q_bo = Q_manual

# epochs = 35
# num_patches = 20
# BS_init = 40
# BS_val = 10

# Get initial lap_time

BS_test = 1

# This sampling should bring always the same set of initial states
x0_lap = utils_new.sample_init_test(BS_test, true_dx, sn=0).numpy()

x0_lap_manual = x0_lap[:,:6]

finish_list = np.zeros((BS_test,))
lap_time_list = np.zeros((BS_test,))

for b in range(BS_test):
    finished = 0
    crashed = 0
    steps = 0
    max_steps=500

    x0_b_manual = x0_lap_manual[0].copy()
    x_manual_full_H = x0_b_manual.reshape(-1,1)

    while finished==0 and crashed==0:
        q_lap_manual_casadi = Q_manual_H[:,idx_to_casadi].T
        p_lap_manual_casadi = p_manual_H[:,idx_to_casadi].T

        x_b_manual, u_b_manual = utils_new.solve_casadi(
            q_lap_manual_casadi, p_lap_manual_casadi,
            x0_b_manual, dx, du, control_H)

        x0_b_manual = x_b_manual[1]
        x_manual_full_H = np.append(x_manual_full_H, x0_b_manual.reshape(-1,1), axis=1)
        if steps == 0:
            u_manual_full_H = u_b_manual[0].reshape(-1,1)
        else:
            u_manual_full_H = np.append(u_manual_full_H, u_b_manual[0].reshape(-1,1), axis=1)

        if x0_b_manual[0]>track_coord[2].max().numpy()/2:
            finished=1

        if x0_b_manual[1]>bound_d_casadi+0.001 or x0_b_manual[1]<-bound_d_casadi-0.001 or steps>max_steps:
            crashed=1

        steps = steps+1

    lap_time_H = dt*steps

    finish_list[b] = finished
    lap_time_list[b] = lap_time_H

    print(f'Manual extended mpc_H = {mpc_H}, lap time: {lap_time_H}')


finish_list = np.zeros((bo_iter+warm_start,))
lap_time_list = np.zeros((bo_iter+warm_start,))

for b in range(warm_start):
    finished = 0
    crashed = 0
    steps = 0
    max_steps=500

    p_bo_add[idx_to_learned_param] = bo_grid[samples[b]]

    p_bo_app = p_bo_base + p_bo_add

    p_bo = np.repeat(np.expand_dims(p_bo_app, 0), mpc_T, 0)

    x0_b_manual = x0_lap_manual[0].copy()
    x_manual_full = x0_b_manual.reshape(-1,1)

    while finished==0 and crashed==0:
        q_lap_manual_casadi = Q_bo[:,idx_to_casadi].T
        p_lap_manual_casadi = p_bo[:,idx_to_casadi].T

        x_b_manual, u_b_manual = utils_new.solve_casadi(
            q_lap_manual_casadi, p_lap_manual_casadi,
            x0_b_manual, dx, du, control)

        x0_b_manual = x_b_manual[1]
        x_manual_full = np.append(x_manual_full, x0_b_manual.reshape(-1,1), axis=1)
        if steps == 0:
            u_manual_full =  u_b_manual[0].reshape(-1,1)
        else:
            u_manual_full = np.append(u_manual_full, u_b_manual[0].reshape(-1,1), axis=1)

        if x0_b_manual[0]>track_coord[2].max().numpy()/2:
            finished=1

        if x0_b_manual[1]>bound_d_casadi+0.001 or x0_b_manual[1]<-bound_d_casadi-0.001 or steps>max_steps:
            crashed=1

        steps = steps+1

    lap_time = dt*steps

    finish_list[b] = finished
    lap_time_list[b] = lap_time

    loss_length = min(np.shape(x_manual_full)[1],np.shape(x_manual_full_H)[1])
    print(np.shape(x_manual_full))
    print(np.shape(x_manual_full_H))

    loss_dsigma = ((x_manual_full[5,:loss_length] - x_manual_full_H[5,:loss_length])**2).sum(0).mean()
    loss_d = ((x_manual_full[1,:loss_length] - x_manual_full_H[1,:loss_length])**2).sum(0).mean()
    loss_phi = ((x_manual_full[2,:loss_length] - x_manual_full_H[2,:loss_length])**2).sum(0).mean()
    loss_v = ((x_manual_full[3,:loss_length] - x_manual_full_H[3,:loss_length])**2).sum(0).mean()

    loss_a = ((u_manual_full[0,:loss_length-1] - u_manual_full_H[0,:loss_length-1])**2).sum(0).mean()
    loss_delta = ((u_manual_full[1,:loss_length-1] - u_manual_full_H[1,:loss_length-1])**2).sum(0).mean()

    loss = 100*loss_dsigma + 100*loss_d + loss_phi + 0.01*loss_a + loss_delta + crashed*10

    loss = loss.reshape(1)

    if b == 0:
        losses = loss
    else:
        losses = np.append(losses,loss,axis=0)

    print(f'Manual mpc_T = {mpc_T}, lap time: {lap_time}')
    print(f'Manual mpc_T = {mpc_T}, loss: {loss}')
    print(f'Manual mpc_T = {mpc_T}, loss length: {loss_length}')



gpr_fit = gpr.fit(bo_grid[samples],losses)

mean_fit, std_fit = gpr.predict(bo_grid, return_std=True)

accq_fun = mean_fit - beta*std_fit

# find minimum of accquisition function to find next sampling point
sampling_idx = np.argmin(accq_fun).reshape(1)

samples = np.append(samples,sampling_idx,axis=0)

# and this is now something we do in a loop again

for i in range(bo_iter):
    finished = 0
    crashed = 0
    steps = 0
    max_steps=500

    p_bo_add[idx_to_learned_param] = bo_grid[sampling_idx]

    p_bo_app = p_bo_base + p_bo_add

    p_bo = np.repeat(np.expand_dims(p_bo_app, 0), mpc_T, 0)

    x0_b_manual = x0_lap_manual[0].copy()
    x_manual_full = x0_b_manual.reshape(-1,1)

    while finished==0 and crashed==0:
        q_lap_manual_casadi = Q_bo[:,idx_to_casadi].T
        p_lap_manual_casadi = p_bo[:,idx_to_casadi].T

        x_b_manual, u_b_manual = utils_new.solve_casadi(
            q_lap_manual_casadi, p_lap_manual_casadi,
            x0_b_manual, dx, du, control)

        x0_b_manual = x_b_manual[1]
        x_manual_full = np.append(x_manual_full, x0_b_manual.reshape(-1,1), axis=1)
        if steps == 0:
            u_manual_full =  u_b_manual[0].reshape(-1,1)
        else:
            u_manual_full = np.append(u_manual_full, u_b_manual[0].reshape(-1,1), axis=1)

        if x0_b_manual[0]>track_coord[2].max().numpy()/2:
            finished=1

        if x0_b_manual[1]>bound_d_casadi+0.001 or x0_b_manual[1]<-bound_d_casadi-0.001 or steps>max_steps:
            crashed=1

        steps = steps+1

    lap_time = dt*steps

    finish_list[warm_start+b] = finished
    lap_time_list[warm_start+b] = lap_time

    loss_length = min(np.shape(x_manual_full)[1],np.shape(x_manual_full_H)[1])
    print(np.shape(x_manual_full))
    print(np.shape(x_manual_full_H))

    loss_dsigma = ((x_manual_full[5,:loss_length] - x_manual_full_H[5,:loss_length])**2).sum(0).mean()
    loss_d = ((x_manual_full[1,:loss_length] - x_manual_full_H[1,:loss_length])**2).sum(0).mean()
    loss_phi = ((x_manual_full[2,:loss_length] - x_manual_full_H[2,:loss_length])**2).sum(0).mean()
    loss_v = ((x_manual_full[3,:loss_length] - x_manual_full_H[3,:loss_length])**2).sum(0).mean()

    loss_a = ((u_manual_full[0,:loss_length-1] - u_manual_full_H[0,:loss_length-1])**2).sum(0).mean()
    loss_delta = ((u_manual_full[1,:loss_length-1] - u_manual_full_H[1,:loss_length-1])**2).sum(0).mean()

    loss = 100*loss_dsigma + 100*loss_d + loss_phi + 0.01*loss_a + loss_delta + crashed*10

    loss = loss.reshape(1)

    losses = np.append(losses,loss,axis=0)

    print(f'Manual mpc_T = {mpc_T}, lap time: {lap_time}')
    print(f'Manual mpc_T = {mpc_T}, loss: {loss}')
    print(f'Manual mpc_T = {mpc_T}, loss length: {loss_length}')

    gpr_fit = gpr.fit(bo_grid[samples],losses)

    mean_fit, std_fit = gpr.predict(bo_grid, return_std=True)

    accq_fun = mean_fit - beta*std_fit

    # find minimum of accquisition function to find next sampling point
    sampling_idx = np.argmin(accq_fun).reshape(1)

    samples = np.append(samples,sampling_idx,axis=0)
